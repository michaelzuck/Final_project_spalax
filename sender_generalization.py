
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import OneClassSVM
from scipy.stats import skew, kurtosis
from collections import Counter, defaultdict


##############################################
# Common Functions (Processing, Feature Extraction & Augmentation)
##############################################

def butter_lowpass(cutoff, fs, order=5):
    """Create a Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    """Apply a low-pass filter on the data."""
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def process_signal(data, sample_rate, cutoff):
    """Process the raw signal using a low-pass filter."""
    return lowpass_filter(data, cutoff, sample_rate)


def extract_features(signal, sample_rate):
    """
    Extract various time-domain and spectral features from the signal.
    Features include: RMS, zero-crossing rate, energy, min/max, mean, variance,
    skewness, kurtosis, FFT-based metrics, etc.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    energy = np.sum(signal ** 2)
    min_val = np.min(signal)
    max_val = np.max(signal)
    mean_val = np.mean(signal)
    variance = np.var(signal)
    skewness_val = skew(signal)
    kurt_val = kurtosis(signal)
    peak_amplitude = np.max(np.abs(signal))
    crest_factor = peak_amplitude / rms if rms != 0 else 0

    n = len(signal)
    freq = fftfreq(n, d=1 / sample_rate)
    fft_values = fft(signal)
    power = np.sum(np.abs(fft_values) ** 2) / n
    dominant_freq = np.abs(freq[np.argmax(np.abs(fft_values))])
    power_spectrum = np.abs(fft_values) ** 2
    spectral_entropy = -np.sum((power_spectrum / np.sum(power_spectrum)) *
                               np.log2(power_spectrum / np.sum(power_spectrum) + 1e-9))
    spectral_centroid = (
        np.sum(freq * power_spectrum) / np.sum(power_spectrum)
        if np.sum(power_spectrum) != 0 else 0
    )
    spectral_spread = (
        np.sqrt(np.sum((freq - spectral_centroid) ** 2 * power_spectrum) / np.sum(power_spectrum))
        if np.sum(power_spectrum) != 0 else 0
    )
    spectral_flatness = np.exp(np.mean(np.log(power_spectrum + 1e-9))) / np.mean(power_spectrum + 1e-9)
    spectral_rolloff = np.sum(power_spectrum.cumsum() <= 0.85 * power_spectrum.sum())
    harmonics = np.sum(np.abs(fft_values[freq > 0.1]))
    noise = np.sum(np.abs(fft_values[freq <= 0.1]))
    hnr = 10 * np.log10(harmonics / (noise + 1e-9)) if noise != 0 else 0
    peak_to_peak = max_val - min_val

    return [
        rms, zcr, power, dominant_freq, energy, min_val, max_val, mean_val,
        variance, spectral_entropy, peak_to_peak, skewness_val, kurt_val,
        peak_amplitude, crest_factor, spectral_centroid, spectral_spread,
        spectral_flatness, spectral_rolloff, hnr
    ]


def augment_signal(signal, sample_rate):
    """
    Perform simple data augmentation on a signal:
      - Time shift: random time shift of the signal.
      - Adding noise: adding a small amount of noise.
    """
    max_shift = int(0.1 * len(signal))
    shift = np.random.randint(-max_shift, max_shift)
    augmented = np.roll(signal, shift)
    noise = np.random.normal(0, 0.005 * np.max(np.abs(signal)), size=signal.shape)
    augmented = augmented + noise
    return augmented


##############################################
# Sender Functions (Data Preparation & Training)
##############################################

def get_sender_label(folder_name):
    """
    Extract the sender label from folder names like:
    'BMR3_vs_BMR4_123'.

    We do this by:
      1. Splitting on the substring '_vs_'.
      2. Taking the first part (e.g., 'BMR3').
      3. Splitting on '_' if needed and returning the first element.
    """
    parts = folder_name.split('_vs_')
    if len(parts) < 2:
        # Fallback if the folder name doesn't follow the pattern
        return folder_name[:4]

    first_part = parts[0]
    subparts = first_part.split('_')
    return subparts[0]  # 'BMR3'


def process_all_files_sender(root_directory, cutoff):
    """
    Walk through all subdirectories and process each .wav file.
    The sender label is extracted using get_sender_label().
    """
    processed_data = {}
    for root, dirs, files in os.walk(root_directory):
        relative_dir = os.path.relpath(root, root_directory)
        if relative_dir == ".":
            continue
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                sample_rate, data = wavfile.read(file_path)
                if len(data.shape) == 2:
                    data = data.mean(axis=1)
                if np.min(data) != 0:
                    data = data / np.min(data)
                filtered_data = process_signal(data, sample_rate, cutoff)
                processed_data.setdefault(relative_dir, {})[file] = (sample_rate, data, filtered_data)
    return processed_data


def prepare_dataset_sender(filtered_signals_dict, exclude_label, use_augmentation=False):
    """
    Prepare the Sender dataset.
    Exclude samples from subfolders whose sender label equals `exclude_label` in training,
    but include them in testing.

    When use_augmentation=True, automatic balancing is performed using augment_signal() until
    balanced with the group having the maximum count.
    """
    X = []
    y = []
    train_signals = []

    subfolder_names = list(filtered_signals_dict.keys())
    train_subfolders, test_subfolders = train_test_split(subfolder_names, test_size=0.3, random_state=42)

    # Build training set
    for subfolder in train_subfolders:
        sender_label = get_sender_label(subfolder)
        if sender_label == exclude_label:
            continue
        for filename in filtered_signals_dict[subfolder]:
            sample_rate, original_signal, filtered_signal = filtered_signals_dict[subfolder][filename]
            features = extract_features(filtered_signal, sample_rate)
            X.append(features)
            y.append(sender_label)
            train_signals.append((sample_rate, filtered_signal, sender_label))

    # Automatic balancing (augmentation)
    if use_augmentation:
        label_samples = defaultdict(list)
        for sample in train_signals:
            sr, sig, lbl = sample
            label_samples[lbl].append((sr, sig))
        max_count = max(len(samples) for samples in label_samples.values())
        augmented_X = []
        augmented_y = []
        for lbl, samples in label_samples.items():
            current_count = len(samples)
            needed = max_count - current_count
            idx = 0
            while needed > 0:
                sr, sig = samples[idx % len(samples)]
                aug_sig = augment_signal(sig, sr)
                aug_features = extract_features(aug_sig, sr)
                augmented_X.append(aug_features)
                augmented_y.append(lbl)
                needed -= 1
                idx += 1
        X.extend(augmented_X)
        y.extend(augmented_y)
        print("After automatic augmentation, training label counts:", Counter(y))

    # Build test set
    X_test = []
    y_test = []
    for subfolder in test_subfolders:
        sender_label = get_sender_label(subfolder)
        for filename in filtered_signals_dict[subfolder]:
            sample_rate, original_signal, filtered_signal = filtered_signals_dict[subfolder][filename]
            features = extract_features(filtered_signal, sample_rate)
            X_test.append(features)
            y_test.append(sender_label)

    X_train = np.array(X)
    y_train = np.array(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Balancing and scaling
    borderline_smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = borderline_smote.fit_resample(X_train, y_train)
    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_res, y_train_res)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_res, y_test])
    label_encoder.fit(all_labels)
    y_train_enc = label_encoder.transform(y_train_res)
    y_test_enc = label_encoder.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, label_encoder


def train_sender_model(exclude_label, use_augmentation=False):
    """Train the Sender RandomForest model with hyperparameter tuning."""
    root_directory = r"C:\signals"  # Adjust the path as needed
    cutoff = 500.0

    print("=== Sender: Gathering data ===")
    filtered_signals_dict = process_all_files_sender(root_directory, cutoff)
    print(f"[Sender] Total subfolders processed: {len(filtered_signals_dict)}")

    # Display number of examples per label before splitting
    sender_counts = {}
    for subfolder, files_dict in filtered_signals_dict.items():
        sender_label = get_sender_label(subfolder)
        sender_counts[sender_label] = sender_counts.get(sender_label, 0) + len(files_dict)
    print("\n[Sender] Number of signals per sender before splitting:")
    for label, count in sender_counts.items():
        print(f"{label}: {count}")

    print(f"\n=== Sender: Preparing dataset (excluding {exclude_label} from training) ===")
    X_train, X_test, y_train_enc, y_test_enc, label_encoder = prepare_dataset_sender(
        filtered_signals_dict, exclude_label, use_augmentation=use_augmentation
    )

    unique_classes = np.unique(y_train_enc)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_enc)
    cw_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Train RandomForest with hyperparameter tuning
    rf_model = RandomForestClassifier(n_jobs=-1, random_state=42, class_weight=cw_dict)
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    print("\n=== Sender: Training RandomForest model with hyperparameter tuning ===")
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=30,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        error_score='raise'
    )
    random_search.fit(X_train, y_train_enc)
    best_rf_model = random_search.best_estimator_

    print("\n[Sender] Best Hyperparameters:")
    print(random_search.best_params_)

    y_pred = best_rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test_enc, y_pred)
    print(f"\n[Sender] RandomForest Accuracy on Test Set: {rf_accuracy:.2f}")
    print("[Sender] Classification Report (RandomForest only):")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, zero_division=0))

    conf_matrix = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Sender: Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return best_rf_model, X_train, label_encoder, X_test, y_test_enc


##############################################
# One-Class SVM Novelty Detection (שימוש משותף)
##############################################

def train_one_class_svm(X_train, nu=0.05, gamma='scale'):
    """
    Train a One-Class SVM for novelty detection.
    nu: Upper bound on the fraction of training errors.
    gamma: Kernel coefficient.
    """
    ocsvm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
    ocsvm.fit(X_train)
    print(f"[One-Class SVM] Trained with nu={nu}, gamma={gamma}")
    return ocsvm


def combined_predict(rf_model, ocsvm_model, X, label_encoder, confidence_threshold=0.6):
    """
    Combine decisions from One-Class SVM and RandomForest classifier.
      - If SVM predicts -1 (outlier), we label it "Unknown".
      - Otherwise, if RF's max probability < confidence_threshold, label "Unknown".
      - Otherwise, use the RF predicted label.
    """
    ocsvm_preds = ocsvm_model.predict(X)  # 1 = inlier, -1 = outlier
    rf_probs = rf_model.predict_proba(X)
    rf_preds = rf_model.predict(X)

    final_preds = []
    for i in range(len(X)):
        if ocsvm_preds[i] == -1:
            final_preds.append("Unknown")
        else:
            if np.max(rf_probs[i]) < confidence_threshold:
                final_preds.append("Unknown")
            else:
                label = label_encoder.inverse_transform([rf_preds[i]])[0]
                final_preds.append(label)
    return final_preds


##############################################
# Group-Level Novelty Detection for Sender
##############################################

def group_level_novelty_test(rf_model, ocsvm_model, sender_label_encoder, root_directory, cutoff, threshold=0.65):
    """
    For each unique sender label:
      1. Aggregate all signals.
      2. Normalize locally.
      3. Run One-Class SVM for outliers.
      4. If outlier % > threshold, group is "Unknown".
         Otherwise, use majority voting with RF for inliers.
    """
    sender_data = process_all_files_sender(root_directory, cutoff)
    grouped_features = {}

    # Group signals by sender label
    for folder in sender_data:
        label = get_sender_label(folder)
        for filename in sender_data[folder]:
            sample_rate, original_signal, filtered_signal = sender_data[folder][filename]
            feat = extract_features(filtered_signal, sample_rate)
            grouped_features.setdefault(label, []).append(feat)

    group_results = {}
    for label, features_list in grouped_features.items():
        features_array = np.array(features_list)
        if len(features_array) == 0:
            continue

        local_scaler = MinMaxScaler()
        features_scaled = local_scaler.fit_transform(features_array)
        svm_preds = ocsvm_model.predict(features_scaled)  # 1 = inlier, -1 = outlier

        unknown_count = np.sum(svm_preds == -1)
        total_count = len(svm_preds)
        unknown_percentage = unknown_count / total_count

        print(
            f"\nGroup {label}: total={total_count}, unknown_count={unknown_count}, unknown%={unknown_percentage * 100:.2f}%")

        # Simple distribution plot
        mapped_preds = ["Unknown" if x == -1 else "Known" for x in svm_preds]
        unique, counts = np.unique(mapped_preds, return_counts=True)
        pred_dict = dict(zip(unique, counts))

        plt.figure()
        plt.bar(list(pred_dict.keys()), list(pred_dict.values()), color='skyblue')
        plt.xlabel('Prediction')
        plt.ylabel('Count')
        plt.title(f'Prediction Distribution for Group {label} (SVM Results)')
        plt.axhline(y=threshold * total_count, color='r', linestyle='--', label=f'Threshold ({threshold * 100:.0f}%)')
        plt.legend()
        plt.show()

        if unknown_percentage > threshold:
            group_results[label] = "Unknown"
        else:
            # Use RF for inlier samples
            rf_preds = []
            for idx, pred in enumerate(svm_preds):
                if pred == 1:
                    single_pred = rf_model.predict(features_scaled[idx].reshape(1, -1))
                    decoded = sender_label_encoder.inverse_transform(single_pred)[0]
                    rf_preds.append(decoded)
            if len(rf_preds) > 0:
                final_label = Counter(rf_preds).most_common(1)[0][0]
            else:
                final_label = "Unknown"
            group_results[label] = final_label

    return group_results


##############################################
# Combined System Test for Sender
##############################################

def combined_system_test_sender(exclude_label, use_augmentation=False):
    """
    1. Train the Sender model (RF) and the One-Class SVM.
    2. Perform group-level novelty detection.
    3. Run a full test with combined predict (RF + SVM).
    """
    root_directory = r"C:\signals"  # Adjust as needed
    cutoff = 500.0
    threshold = 0.8

    print("========== Combined System Test for Sender ==========")
    print("Step 1: Training Sender Model (RandomForest)")
    rf_model, X_train, sender_label_encoder, X_test, y_test_enc = train_sender_model(
        exclude_label, use_augmentation=use_augmentation
    )

    print("Step 2: Training One-Class SVM for Sender (novelty detection)")
    ocsvm_sender = train_one_class_svm(X_train, nu=0.05, gamma='scale')

    print("\nStep 3: Group-Level Novelty Test")
    group_results = group_level_novelty_test(
        rf_model, ocsvm_sender, sender_label_encoder,
        root_directory, cutoff, threshold
    )

    print("\nGroups detected as unknown:")
    unknown_groups = {g: lbl for g, lbl in group_results.items() if lbl == "Unknown"}
    if unknown_groups:
        for group in unknown_groups:
            print(f"- {group} was classified as Unknown")
    else:
        print("No groups were classified as Unknown.")

    print("\nStep 4: Full Test on All Sender Signals (Combined Prediction)")
    filtered_signals_dict = process_all_files_sender(root_directory, cutoff)
    _, X_test_full, _, y_test_full_enc, label_encoder_full = prepare_dataset_sender(
        filtered_signals_dict, exclude_label, use_augmentation
    )
    y_test_full_decoded = label_encoder_full.inverse_transform(y_test_full_enc)

    # Combined predict
    preds_full = combined_predict(rf_model, ocsvm_sender, X_test_full, sender_label_encoder, confidence_threshold=0.6)

    print(f"\n[Sender Model] Full Test on All Signals (excluding '{exclude_label}' and 'Unknown'):")
    mask = (y_test_full_decoded != exclude_label) & (np.array(preds_full) != "Unknown")
    if np.sum(mask) > 0:
        print("Classification Report (RF on known groups only):")
        print(classification_report(y_test_full_decoded[mask], np.array(preds_full)[mask], zero_division=0))
        conf_matrix_full = confusion_matrix(
            y_test_full_decoded[mask],
            np.array(preds_full)[mask],
            labels=np.unique(y_test_full_decoded[mask])
        )
        disp_full = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_full,
                                           display_labels=np.unique(y_test_full_decoded[mask]))
        disp_full.plot(cmap=plt.cm.Blues)
        plt.title('Sender: Confusion Matrix on Full Test Set (filtered)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    else:
        print("No samples available for RF evaluation (all were excluded or unknown).")

    print("\n========== Combined System Test for Sender Completed ==========")


##############################################
# Main Function (Optional CLI)
##############################################

def main():
    root_directory = r"C:\signals"
    cutoff = 500.0

    filtered_signals_dict = process_all_files_sender(root_directory, cutoff)
    available_labels = sorted(set([get_sender_label(folder) for folder in filtered_signals_dict.keys()]))

    print("Available sender groups for exclusion:")
    for label in available_labels:
        print(f"- {label}")

    exclude_label = input("Please enter the sender label you wish to exclude from training: ").strip()
    if exclude_label not in available_labels:
        print(f"Warning: The label '{exclude_label}' is not among the available groups: {available_labels}")

    use_augmentation = True
    combined_system_test_sender(exclude_label, use_augmentation=use_augmentation)


if __name__ == "__main__":
    main()
