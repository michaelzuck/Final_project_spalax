
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import skew, kurtosis
from collections import Counter


# ----------------------------------------------------
# Signal Processing Functions
# ----------------------------------------------------
def butter_lowpass(cutoff, fs, order=5):
    """
    Calculate the lowpass filter coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply the lowpass filter to the data.
    """
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

def process_signal(data, sample_rate, cutoff):
    """
    Process the signal by applying the lowpass filter.
    """
    return lowpass_filter(data, cutoff, sample_rate)


# ----------------------------------------------------
# Feature Extraction Function (20 features)
# ----------------------------------------------------
def extract_features(signal, sample_rate):
    """
    Extract 20 features from an audio signal:
    1) RMS
    2) ZCR
    3) Power
    4) Dominant Frequency
    5) Energy
    6) Minimum Value
    7) Maximum Value
    8) Mean Value
    9) Variance
    10) Spectral Entropy
    11) Spectral Rolloff
    12) Skewness
    13) Kurtosis
    14) Peak Amplitude
    15) Crest Factor
    16) Spectral Centroid
    17) Spectral Spread
    18) Spectral Flatness
    19) HNR (Harmonics-to-Noise Ratio)
    20) Energy (repeated to match the original 20 features)
    """
    if len(signal) == 0:
        return np.zeros(20, dtype=float)

    # Time-domain features
    rms = np.sqrt(np.mean(signal ** 2))
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / (len(signal) - 1) if len(signal) > 1 else 0
    energy = np.sum(signal ** 2)
    min_val = np.min(signal)
    max_val = np.max(signal)
    mean_val = np.mean(signal)
    variance = np.var(signal)
    skewness_val = skew(signal)
    kurt_value = kurtosis(signal)
    peak_amplitude = np.max(np.abs(signal))
    crest_factor = peak_amplitude / rms if rms != 0 else 0

    # Frequency-domain features
    n = len(signal)
    freq = fftfreq(n, d=1 / sample_rate)
    fft_values = fft(signal)
    power_spectrum = np.abs(fft_values) ** 2
    power = np.sum(power_spectrum) / n

    # Dominant Frequency
    dominant_freq = np.abs(freq[np.argmax(np.abs(fft_values))]) if n > 1 else 0

    total_power = np.sum(power_spectrum)
    if total_power > 0:
        ps_norm = power_spectrum / total_power
        spectral_entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-9))
    else:
        spectral_entropy = 0

    spectral_centroid = (np.sum(freq * power_spectrum) / total_power) if total_power > 0 else 0
    spectral_spread = np.sqrt(
        np.sum((freq - spectral_centroid) ** 2 * power_spectrum) / total_power
    ) if total_power > 0 else 0
    spectral_flatness = np.exp(np.mean(np.log(power_spectrum + 1e-9))) / np.mean(power_spectrum + 1e-9)

    # Calculate the spectral rolloff (85% threshold)
    cumulative_sum = np.cumsum(power_spectrum)
    rolloff_threshold = 0.85 * total_power
    rolloff_indices = np.where(cumulative_sum >= rolloff_threshold)[0]
    spectral_rolloff = rolloff_indices[0] if len(rolloff_indices) > 0 else 0

    # Harmonics vs. Noise
    harmonics = np.sum(np.abs(fft_values[freq > 0.1]))
    noise = np.sum(np.abs(fft_values[freq <= 0.1]))
    hnr = 10 * np.log10((harmonics / (noise + 1e-9)) + 1e-9) if noise != 0 else 0

    return [
        rms, zcr, power, dominant_freq, energy, min_val, max_val, mean_val, variance,
        spectral_entropy, spectral_rolloff, skewness_val, kurt_value, peak_amplitude,
        crest_factor, spectral_centroid, spectral_spread, spectral_flatness, hnr, energy
    ]


# ----------------------------------------------------
# Parse Folder Name to Extract Recipient Label
# ----------------------------------------------------
def parse_recipient(folder_name):
    """
    Extract the recipient label from the folder name.
    For example, given "BMR2_vs_BMR3_28", it returns "BMR3".
    """
    parts = folder_name.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Folder name '{folder_name}' does not match the expected pattern 'X_vs_Y...'")
    recipient = parts[1].split("_")[0]
    return recipient


# ----------------------------------------------------
# Gather Data from Subfolders
# ----------------------------------------------------
def process_all_files(root_directory, cutoff):
    """
    Walk through the root directory and its subfolders.
    Return a dictionary: {subfolder: {filename: (sample_rate, original_data, filtered_data)}}
    Only subfolders that follow the expected naming pattern (for recipient extraction) are processed.
    """
    processed_data = {}
    for root, dirs, files in os.walk(root_directory):
        relative_dir = os.path.relpath(root, root_directory)
        # Skip the root directory itself
        if relative_dir == ".":
            continue

        # Attempt to parse the recipient label from the folder name
        try:
            _ = parse_recipient(relative_dir)
        except ValueError:
            continue

        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    sample_rate, data = wavfile.read(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue

                # Convert stereo to mono by averaging channels, if necessary
                if len(data.shape) == 2:
                    data = data.mean(axis=1)

                # Simple normalization by dividing by the minimum value (if not zero)
                min_data = np.min(data)
                if min_data != 0:
                    data = data / min_data

                # Apply lowpass filter
                filtered_data = process_signal(data, sample_rate, cutoff)

                if relative_dir not in processed_data:
                    processed_data[relative_dir] = {}
                processed_data[relative_dir][file] = (sample_rate, data, filtered_data)

    return processed_data


# ----------------------------------------------------
# Prepare Dataset by Subfolder (Maintain Folder Integrity)
# ----------------------------------------------------
def prepare_dataset_by_subfolder(filtered_signals_dict, train_ratio=0.7):
    """
    1) Each subfolder remains as a single unit (files in the same folder are not split between train and test).
    2) The label is extracted using parse_recipient(subfolder).
    3) Aim for ~70% of the files per recipient in train and ~30% in test.
    """
    # Group subfolders by recipient label
    label_dict = {}
    for subfolder, files_dict in filtered_signals_dict.items():
        label = parse_recipient(subfolder)
        count = len(files_dict)
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append((subfolder, count))

    train_subfolders = []
    test_subfolders = []

    # For each recipient, shuffle their subfolders and split based on file count
    for label, subfolder_list in label_dict.items():
        np.random.shuffle(subfolder_list)
        total_count = sum(cnt for _, cnt in subfolder_list)
        target_train = train_ratio * total_count
        current_train = 0
        for subfolder, cnt in subfolder_list:
            if current_train < target_train:
                train_subfolders.append(subfolder)
                current_train += cnt
            else:
                test_subfolders.append(subfolder)

    # Build training and testing datasets
    X_train, y_train, filenames_train = [], [], []
    X_test, y_test, filenames_test = [], [], []

    for subfolder in train_subfolders:
        for file, data_tuple in filtered_signals_dict[subfolder].items():
            sample_rate, original_signal, filtered_signal = data_tuple
            features = extract_features(filtered_signal, sample_rate)
            X_train.append(features)
            label = parse_recipient(subfolder)
            y_train.append(label)
            filenames_train.append((subfolder, file))

    for subfolder in test_subfolders:
        for file, data_tuple in filtered_signals_dict[subfolder].items():
            sample_rate, original_signal, filtered_signal = data_tuple
            features = extract_features(filtered_signal, sample_rate)
            X_test.append(features)
            label = parse_recipient(subfolder)
            y_test.append(label)
            filenames_test.append((subfolder, file))

    return (np.array(X_train), np.array(X_test),
            np.array(y_train), np.array(y_test),
            filenames_train, filenames_test)


# ----------------------------------------------------
# Main Model Training Function
# ----------------------------------------------------
def train_model():
    root_directory = r"C:\signals"  # Update this path as needed
    cutoff = 500.0

    print("Gathering data from subfolders...")
    filtered_signals_dict = process_all_files(root_directory, cutoff)
    print(f"Total subfolders processed: {len(filtered_signals_dict)}")

    # Print the number of files per recipient (label) before splitting
    label_counts = {}
    for subfolder, files_dict in filtered_signals_dict.items():
        label = parse_recipient(subfolder)
        label_counts[label] = label_counts.get(label, 0) + len(files_dict)

    print("\nNumber of files per recipient (Label) before splitting:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nSplitting the dataset by subfolder (~70% training, ~30% testing)...")
    X_train, X_test, y_train, y_test, f_train, f_test = prepare_dataset_by_subfolder(
        filtered_signals_dict, train_ratio=0.7
    )

    # Oversample the training set using BorderlineSMOTE then ADASYN
    borderline_smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = borderline_smote.fit_resample(X_train, y_train)

    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_res, y_train_res)

    # Normalize the data with MinMaxScaler
    scaler = MinMaxScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels to numeric values
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_res, y_test])
    label_encoder.fit(all_labels)
    y_train_enc = label_encoder.transform(y_train_res)
    y_test_enc = label_encoder.transform(y_test)

    # Compute class weights for imbalanced classes
    unique_classes = np.unique(y_train_enc)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_enc)
    cw_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Define the Random Forest model
    rf_model = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        class_weight=cw_dict
    )

    # Hyperparameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    print("\nTraining Random Forest model with hyperparameter tuning (RandomizedSearchCV)...")
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        error_score='raise'
    )
    random_search.fit(X_train_res, y_train_enc)
    best_rf_model = random_search.best_estimator_
    print("\nBest Hyperparameters:")
    print(random_search.best_params_)

    # Evaluate the model on the test set
    y_pred = best_rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\nTest Set Accuracy: {acc:.3f}")

    # Print full classification report (Precision, Recall, F1, Support)
    class_report = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("\nFull Classification Report:")
    print(class_report)

    # Display the confusion matrix
    conf_matrix = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ----------------------------------------------------
    # Plot Bar Chart for Precision, Recall, and F1-Score (Including Macro and Weighted Avg)
    # ----------------------------------------------------
    report_dict = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    # Exclude the "accuracy" key since it is a single number
    plot_labels = [k for k in report_dict.keys() if k != 'accuracy']
    precisions = [report_dict[label]['precision'] for label in plot_labels]
    recalls    = [report_dict[label]['recall']    for label in plot_labels]
    f1s        = [report_dict[label]['f1-score']  for label in plot_labels]

    x = np.arange(len(plot_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width, precisions, width, label='Precision')
    rects2 = plt.bar(x,         recalls,    width, label='Recall')
    rects3 = plt.bar(x + width, f1s,        width, label='F1-Score')

    plt.title('Precision, Recall, and F1-Score by Class (Including Macro and Weighted Avg)')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.ylim(0, 1.1)

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.xticks(x, plot_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_rf_model, X_train_res, y_train_enc


# If running this file directly, train the model
if __name__ == "__main__":
    trained_model, X_train_global, y_train_global = train_model()
