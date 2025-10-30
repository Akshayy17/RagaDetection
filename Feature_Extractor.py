import os
import csv
import numpy as np
import librosa
from tqdm import tqdm
from collections import Counter
from scipy.stats import skew, kurtosis, entropy

# ========== CONFIGURATION ==========
INPUT_FOLDER = './Bhaatkhande_collection_1894'  # audio folder
OUTPUT_CSV = "result_24_10_25_Full.csv"
MAX_SONGS = 100000000000
TOP_N = 12
SUPPORTED_EXTENSIONS = [".mp3", ".wav"]
SR = 44100
DIVISIONS = 3  # Spectrogram shape info
N_MELS = 1700
GLOBAL_THRESH = 10
COLUMN_THRESH = 10
RATIO_LIST = [1, 2, 3, 4]
BACKGROUND_VALUE = -80
TOP_FREQ_PERCENTAGE = 15

BIN_EDGES = np.array([
    110, 116.39891589, 123.29601869, 130.6662243,
    138.49894393, 146.80959813, 155.58328037, 164.84548598,
    174.57431776, 184.8136729, 195.8082243, 207.45629907,
    220
])

BIN_LABELS = ['2A', '2A#', '2B', '3C', '3C#', '3D',
              '3D#', '3E', '3F', '3F#', '3G', '3G#']

# ========== FUNCTION: filename parser ==========
def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    serial = parts[0]
    thaat = parts[1]

    raga_terms = []
    i = 2
    while i < len(parts) and '(' not in parts[i]:
        raga_terms.append(parts[i])
        i += 1
    raga = "_".join(raga_terms)

    taal = parts[i] if i < len(parts) else ""
    i += 1
    song_title = "_".join(parts[i:]) if i < len(parts) else ""
    return [serial, thaat, raga, taal, song_title]

# ========== AUDIO PROCESSING FUNCTIONS ==========
def normalize_audio(y):
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
    return y

def compute_features(y, sr, n_fft, hop_length, n_mels, global_thresh, per_column_thresh):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, window='hann'
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    global_thresh_val = np.percentile(mel_spec_db, 100 - global_thresh)
    mel_spec_db[mel_spec_db < global_thresh_val] = BACKGROUND_VALUE

    for i in range(mel_spec_db.shape[1]):
        col_thresh = np.percentile(mel_spec_db[:, i], 100 - per_column_thresh)
        mel_spec_db[mel_spec_db[:, i] < col_thresh, i] = BACKGROUND_VALUE

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
    cutoff_bin = np.argmax(mel_freqs > 2500)
    mel_spec_db[cutoff_bin:, :] = BACKGROUND_VALUE

    freq_bins = np.fft.rfftfreq(n_fft, d=1 / sr)
    low_cut_bin = np.argmin(freq_bins < 150)
    mel_spec_db[:low_cut_bin, :] = BACKGROUND_VALUE
    return mel_spec_db

def modify_mel_spec_top_to_bottom(mel_spec_db):
    modified_spec = mel_spec_db.copy()
    rows, cols = mel_spec_db.shape
    for i in range(cols):
        for j in range(1, rows):
            if modified_spec[j - 1, i] < modified_spec[j, i]:
                modified_spec[j - 1, i] = BACKGROUND_VALUE
    return modified_spec

def modify_mel_spec_bottom_to_top(mel_spec_db):
    modified_spec = mel_spec_db.copy()
    rows, cols = mel_spec_db.shape
    for i in range(cols):
        for j in range(rows - 2, -1, -1):
            if modified_spec[j + 1, i] < modified_spec[j, i]:
                modified_spec[j + 1, i] = BACKGROUND_VALUE
    return modified_spec

def custom_and_operation(spec1, spec2):
    background_mask = (spec1 == BACKGROUND_VALUE) & (spec2 == BACKGROUND_VALUE)
    averaged_spec = (spec1 + spec2)
    return np.where(background_mask, BACKGROUND_VALUE, averaged_spec)

def fold_frequency(freq):
    if freq == 0:
        return 0
    while freq < 110:
        freq *= 2
    while freq >= 220:
        freq /= 2
    return freq

def find_best_frequencies(spectrogram, sr, ratio_list):
    num_freq_bins, num_time_frames = spectrogram.shape
    freqs = librosa.mel_frequencies(n_mels=num_freq_bins, fmin=0, fmax=sr / 2)
    overtone_matrix = np.full((len(ratio_list), num_time_frames, 2),
                              [0, BACKGROUND_VALUE], dtype=np.float32)

    for col in range(num_time_frames):
        valid_indices = np.where(spectrogram[:, col] > BACKGROUND_VALUE)[0]
        if len(valid_indices) < len(ratio_list):
            continue

        freq_values = freqs[valid_indices]
        intensity_values = spectrogram[valid_indices, col]
        best_sequence, best_indices, best_intensity_sum = None, None, -np.inf

        for i in range(len(freq_values)):
            x1 = freq_values[i]
            expected_freqs = np.array([x1 * r for r in ratio_list])
            tolerance = expected_freqs * 0.02

            matches, intensities, match_indices = [], [], []
            for ef, tol in zip(expected_freqs, tolerance):
                mask = np.abs(freq_values - ef) <= tol
                if not np.any(mask):
                    break
                best_idx = np.argmax(intensity_values[mask])
                matches.append(float(freq_values[mask][best_idx]))
                intensities.append(
                    float(intensity_values[mask][best_idx]) * float(freq_values[mask][best_idx]) ** 0.1
                )
                match_indices.append(np.where(mask)[0][best_idx])

            if len(matches) == len(ratio_list):
                total_intensity = sum(intensities)
                if total_intensity > best_intensity_sum:
                    best_sequence, best_indices, best_intensity_sum = matches, match_indices, total_intensity

        if best_sequence:
            for i, f in enumerate(best_sequence):
                original_idx = valid_indices[best_indices[i]]
                overtone_matrix[i, col, 0] = freqs[original_idx]
                overtone_matrix[i, col, 1] = spectrogram[original_idx, col]

    return overtone_matrix

# ========== NEW FEATURE EXTRACTION ==========
def compute_additional_features(y, sr, mel_spec_db, bin_counts):
    features = {}

    # --- Basic Statistics ---
    features["mean"] = np.mean(y)
    features["median"] = np.median(y)
    features["std"] = np.std(y)
    features["skew"] = skew(y)
    features["kurtosis"] = kurtosis(y)

    # --- Energy and Time-domain ---
    features["energy"] = np.sum(y ** 2)
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))

    # --- Spectral features ---
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
    features["spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr))

    # --- RMSE ---
    features["rmse"] = np.mean(librosa.feature.rms(S=stft))

    # --- Diversity & Entropy in folded bins ---
    probs = np.array(list(bin_counts.values())) / sum(bin_counts.values()) if len(bin_counts) > 0 else [0]
    features["entropy"] = entropy(probs)
    #features["diversity"] = len(bin_counts)

    # --- Intensity spread ---
    non_bg = mel_spec_db[mel_spec_db > BACKGROUND_VALUE]
    features["intensity_spread"] = np.percentile(non_bg, 90) - np.percentile(non_bg, 10) if len(non_bg) > 0 else 0

    return features

# ========== MAIN PROCESS ==========
def process_all_songs(TOP_FREQ_PERCENTAGE=10):
    files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])
    files = files[:min(MAX_SONGS, len(files))]

    if not files:
        print("No audio files found.")
        return

    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # --- write headers ---
        headers = ["serial","thaat","raga","taal","song"]
        headers += BIN_LABELS[:TOP_N]
        extra_features = ["mean","median","std","skew","kurtosis","energy","zcr",
                          "spectral_centroid","spectral_rolloff","spectral_bandwidth",
                          "spectral_contrast","rmse","entropy","diversity","intensity_spread"]
        headers += extra_features
        writer.writerow(headers)

        for filename in tqdm(files, desc="Processing files", unit="file"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            try:
                y, sr = librosa.load(file_path, sr=SR)
                y = normalize_audio(y)

                duration = int(len(y) / sr) + 1
                y = np.pad(y, (0, sr * duration - len(y)), mode='constant')

                n_fft = int(sr / DIVISIONS)
                hop_length = int(n_fft / 5)

                mel_spec_db = compute_features(y, sr, n_fft, hop_length,
                                               N_MELS, GLOBAL_THRESH, COLUMN_THRESH)
                top_mod = modify_mel_spec_top_to_bottom(mel_spec_db)
                bot_mod = modify_mel_spec_bottom_to_top(mel_spec_db)
                mel_and = custom_and_operation(top_mod, bot_mod)
                overtone_matrix = find_best_frequencies(mel_and, sr, RATIO_LIST)

                overtone_first = overtone_matrix[0]
                base = os.path.splitext(filename)[0]
                save_path = os.path.join(INPUT_FOLDER, f"{base}_f0.npy")
                np.save(save_path, overtone_first)

                fundamental_freqs = overtone_first[:, 0]
                fundamental_freqs = fundamental_freqs[fundamental_freqs > 0]

                if len(fundamental_freqs) > 0:
                    folded = np.array([fold_frequency(f) for f in fundamental_freqs])
                    bin_indices = np.digitize(folded, BIN_EDGES)
                    counts = Counter(bin_indices)
                    total_count = sum(counts.values())

                    # top bins as percentages
                    top_bin_values = [ (counts.get(i+1,0)/total_count)*100 if total_count>0 else 0 for i in range(TOP_N) ]

                    # compute extra features
                    feature_values = compute_additional_features(y, sr, mel_spec_db, counts)
                    feature_list = [feature_values[f] for f in extra_features]

                    meta = parse_filename(filename)
                    writer.writerow(meta + top_bin_values + feature_list)

                del y, mel_spec_db, top_mod, bot_mod, mel_and, overtone_matrix, overtone_first

            except Exception as e:
                print(f"✗ Failed {filename}: {e}")

# ========== RUN ==========
if __name__ == "__main__":
    process_all_songs(TOP_FREQ_PERCENTAGE)
