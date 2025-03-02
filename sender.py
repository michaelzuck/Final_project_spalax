
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


# -------------------------------
# Signal Processing Functions
# -------------------------------
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def process_signal(data, sample_rate, cutoff):
    return lowpass_filter(data, cutoff, sample_rate)


# -------------------------------
# Feature Extraction Function (20 features)
# -------------------------------
def extract_features(signal, sample_rate):
    rms = np.sqrt(np.mean(signal ** 2))
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    energy = np.sum(signal ** 2)
    min_val = np.min(signal)
    max_val = np.max(signal)
    mean_val = np.mean(signal)
    variance = np.var(signal)
    skewness = skew(signal)
    kurt_val = kurtosis(signal)
    peak_amplitude = np.max(np.abs(signal))
    crest_factor = peak_amplitude / rms if rms != 0 else 0

    n = len(signal)
    freq = fftfreq(n, d=1 / sample_rate)
    fft_values = fft(signal)
    power = np.sum(np.abs(fft_values) ** 2) / n
    dominant_freq = np.abs(freq[np.argmax(np.abs(fft_values))])

    power_spectrum = np.abs(fft_values) ** 2
    spectral_entropy = -np.sum(
        (power_spectrum / np.sum(power_spectrum)) * np.log2(power_spectrum / np.sum(power_spectrum) + 1e-9)
    )
    spectral_centroid = (
        np.sum(freq * power_spectrum) / np.sum(power_spectrum)
        if np.sum(power_spectrum) != 0 else 0
    )
    spectral_spread = (
        np.sqrt(
            np.sum((freq - spectral_centroid) ** 2 * power_spectrum) / np.sum(power_spectrum)
        ) if np.sum(power_spectrum) != 0 else 0
    )
    spectral_flatness = np.exp(np.mean(np.log(power_spectrum + 1e-9))) / np.mean(power_spectrum + 1e-9)
    spectral_rolloff = np.sum(power_spectrum.cumsum() <= 0.85 * power_spectrum.sum())

    harmonics = np.sum(np.abs(fft_values[freq > 0.1]))
    noise = np.sum(np.abs(fft_values[freq <= 0.1]))
    hnr = 10 * np.log10(harmonics / (noise + 1e-9)) if noise != 0 else 0

    return [
        rms, zcr, power, dominant_freq, energy, min_val, max_val, mean_val, variance, spectral_entropy,
        spectral_rolloff, skewness, kurt_val, peak_amplitude, crest_factor, spectral_centroid,
        spectral_spread, spectral_flatness, hnr, energy
    ]


# -------------------------------
# Parsing Folder Names to Extract Sender Label
# -------------------------------
def parse_sender(folder_name):
    # Extract the sender label from the subfolder (first 4 chars)
    return folder_name[:4]


# -------------------------------
# Gather Data from Files
# -------------------------------
def process_all_files(root_directory, cutoff):
    processed_data = {}
    for root, dirs, files in os.walk(root_directory):
        relative_dir = os.path.relpath(root, root_directory)
        if relative_dir == ".":
            continue

        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                sample_rate, data = wavfile.read(file_path)

                # If stereo, convert to mono by averaging channels
                if len(data.shape) == 2:
                    data = data.mean(axis=1)

                # Simple normalization by dividing by the minimum if not zero
                min_data = np.min(data)
                if min_data != 0:
                    data = data / min_data

                filtered_data = process_signal(data, sample_rate, cutoff)

                if relative_dir not in processed_data:
                    processed_data[relative_dir] = {}
                processed_data[relative_dir][file] = (sample_rate, data, filtered_data)
    return processed_data


# -------------------------------
# Prepare Dataset by Sender
# -------------------------------
def prepare_dataset_by_sender(filtered_signals_dict, train_ratio=0.7):
    """
    1) Keep all signals from the same subfolder together.
    2) For each sender (subfolder[:4]), aim for ~70% train, ~30% test.
    """
    # Group subfolders by sender
    sender_dict = {}
    for subfolder, files_dict in filtered_signals_dict.items():
        sender = parse_sender(subfolder)
        count = len(files_dict)
        if sender not in sender_dict:
            sender_dict[sender] = []
        sender_dict[sender].append((subfolder, count))

    train_subfolders = []
    test_subfolders = []
    for sender, subfolder_list in sender_dict.items():
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

    X_train, y_train, filenames_train = [], [], []
    X_test, y_test, filenames_test = [], [], []
    for subfolder in train_subfolders:
        for file, data_tuple in filtered_signals_dict[subfolder].items():
            sample_rate, original_signal, filtered_signal = data_tuple
            features = extract_features(filtered_signal, sample_rate)
            X_train.append(features)
            label = parse_sender(subfolder)
            y_train.append(label)
            filenames_train.append((subfolder, file))
    for subfolder in test_subfolders:
        for file, data_tuple in filtered_signals_dict[subfolder].items():
            sample_rate, original_signal, filtered_signal = data_tuple
            features = extract_features(filtered_signal, sample_rate)
            X_test.append(features)
            label = parse_sender(subfolder)
            y_test.append(label)
            filenames_test.append((subfolder, file))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, filenames_train, filenames_test


# -------------------------------
# Train Model Function
# -------------------------------
def train_model():
    root_directory = r"C:\signals"  # <-- Update path as needed
    cutoff = 500.0

    print("Gathering data...")
    filtered_signals_dict = process_all_files(root_directory, cutoff)
    print(f"Total subfolders processed: {len(filtered_signals_dict)}")

    # Print number of samples for each sender before splitting
    sender_counts = {}
    for subfolder, files_dict in filtered_signals_dict.items():
        sender = parse_sender(subfolder)
        sender_counts[sender] = sender_counts.get(sender, 0) + len(files_dict)
    print("\nNumber of signals per sender before splitting:")
    for sender, count in sender_counts.items():
        print(f"{sender}: {count}")

    print("\nPreparing dataset by sender (~70% train, ~30% test)...")
    X_train, X_test, y_train, y_test, f_train, f_test = prepare_dataset_by_sender(
        filtered_signals_dict, train_ratio=0.7
    )

    # Balance the training set
    borderline_smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = borderline_smote.fit_resample(X_train, y_train)

    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_res, y_train_res)

    # Scale the data
    scaler = MinMaxScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_res, y_test])
    label_encoder.fit(all_labels)
    y_train_enc = label_encoder.transform(y_train_res)
    y_test_enc = label_encoder.transform(y_test)

    # Compute class weights
    unique_classes = np.unique(y_train_enc)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_enc)
    cw_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Define Random Forest model
    rf_model = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        class_weight=cw_dict
    )

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    print("\nTraining Random Forest model with hyperparameter tuning...")
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

    # Evaluate on the test set
    y_pred = best_rf_model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\nRandom Forest Accuracy on Test Set: {acc:.3f}")

    # Print full classification report (the table of metrics)
    # This includes Precision, Recall, F1, and Support for each class,
    # as well as the macro avg and weighted avg rows.
    class_report = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("\nRandom Forest Classification Report:")
    print(class_report)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # -------------------------------
    # Plot Precision, Recall, F1 (Including macro avg & weighted avg)
    # -------------------------------
    # classification_report as a dictionary to extract metrics numerically
    report_dict = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    # The keys typically include each class label plus "accuracy", "macro avg", and "weighted avg"
    # We only want classes + "macro avg" + "weighted avg" (exclude "accuracy" because it's a single number)
    plot_labels = [k for k in report_dict.keys() if k != 'accuracy']

    # Extract Precision, Recall, and F1 for each label
    precisions = [report_dict[label]['precision'] for label in plot_labels]
    recalls = [report_dict[label]['recall'] for label in plot_labels]
    f1s = [report_dict[label]['f1-score'] for label in plot_labels]

    x = np.arange(len(plot_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width, precisions, width, label='Precision')
    rects2 = plt.bar(x, recalls, width, label='Recall')
    rects3 = plt.bar(x + width, f1s, width, label='F1-Score')

    plt.title('Precision, Recall, and F1-Score by Class (Including Macro and Weighted Avg)')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.ylim(0, 1.1)  # so labels don't go off the top

    # Place numeric value labels above each bar
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # offset in points
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
