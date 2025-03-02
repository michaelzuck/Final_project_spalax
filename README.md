# Final_project_spalax

Overview
funal_project_spalax is a project designed to process seismic/audio signals, extract relevant features, and classify them by two main categories:

Sender: Classifying signals by which “sender” (source) produced them.
Recipient: Classifying signals by which “recipient” (destination) they were intended for.
This repository contains Python scripts that:

Load .wav audio files (found in the signals directory).
Apply signal-processing techniques (low-pass filtering, feature extraction).
Train machine learning models (RandomForest, One-Class SVM) to classify or detect “unknown” (novelty detection).
Evaluate performance via confusion matrices, classification reports, and permutation tests.
A simple Tkinter-based GUI (combined_gui.py) is also provided to train, test, and visualize results in an integrated environment.

Project Structure

final_project_spalax/
│
├── signals/                          # Directory containing .wav data, organized in subfolders.
│   ├── SenderA_vs_SenderB_01/       # Example subfolder structure
│   ├── SenderC_vs_SenderD_02/
│   └── ... (additional subfolders)
│
├── sender.py                         # Main script for Sender classification (train_model)
├── recepient.py                     # Main script for Recipient classification (train_model)
├── sender_generalization.py          # Advanced generalization script for Sender 
├── recpient_generaliztion.py         # Advanced generalization script for Recipient
├── test_sender.py                    # Permutation test for trained Sender model
├── test_recipient.py                 # Permutation test for trained Recipient model
├── combined_gui.py                   # Tkinter GUI integrating Sender & Recipient workflows
│
└── README.md                         # You're reading it now
Important: The signals directory is expected to contain subfolders for each class (or each combination like BMR2_vs_BMR3_28), where each subfolder has .wav files relevant to that particular class configuration. For example, if you are training a sender model, subfolders might be named in a way that the “sender” can be parsed from the folder name.

Installation & Requirements
Python Version
It is recommended to use Python 3.8+.

Dependencies
Install the required Python libraries (e.g., via pip). You can create a requirements.txt file or install them manually:

pip install numpy scipy matplotlib scikit-learn imbalanced-learn
pip install --upgrade tkinter  # usually included by default on many systems

The main libraries used include:

numpy
scipy
scikit-learn
matplotlib
imbalanced-learn (for SMOTE, ADASYN)
tkinter (for the GUI)
collections, os, io, etc. (standard Python libraries)
Project Setup

Clone or download this repository:
git clone https://github.com/YourUsername/funal_project_spalax.git
Ensure the signals directory is in the same folder as all the .py files.
Populate signals with your .wav data, organized by subfolders that match your desired labeling scheme (see Data Organization).
Data Organization
Place your .wav files inside the signals directory, in subfolders named to indicate the class label. For example:
signals/
└── BMR2_vs_BMR3_28/
     ├── recording_01.wav
     ├── recording_02.wav
     └── ...
└── BMR2_vs_BMR4_10/
     ├── some_file.wav
     └── ...
For Sender Classification:
The code in sender.py attempts to parse a “sender” label from the subfolder name, typically taking the string up to _vs_. E.g., BMR2_vs_BMR3_28 → sender = BMR2.

For Recipient Classification:
The code in recepient.py (and similarly spelled recipient.py if you rename) attempts to parse the “recipient” label from the subfolder name by taking the portion after _vs_.
E.g., BMR2_vs_BMR3_28 → recipient = BMR3.

Please ensure your folder naming follows the patterns expected by the scripts, or adjust the parsing functions accordingly.

Usage
You have multiple ways to use this project:

1. Command-Line Use
Train the Sender Model

python sender.py
This will load data from signals, parse each subfolder as a sender label, split into train/test, apply SMOTE & ADASYN oversampling, and train a RandomForestClassifier.
A confusion matrix and classification report will be displayed in a Matplotlib window.
Final model and train data are returned (trained_model, X_train_global, y_train_global).
Train the Recipient Model

python recepient.py
Similar process, but it parses the “recipient” label from folder names.
Test with Permutation Tests
There are two scripts for permutation-based significance tests:

test_sender.py
test_recipient.py
Each script imports the corresponding train_model() function, trains the model, then performs a permutation test on the training set.
Example:

python test_sender.py
This will run cross-validation on the sender model, shuffle labels multiple times (300 permutations by default), and show a histogram to compute a p-value.

Generalization Scripts

sender_generalization.py
recpient_generaliztion.py
These provide advanced functionality such as excluding certain labels from training (to see if the model can generalize to novel classes), or combining RandomForest with One-Class SVM for novelty detection.
Each script has a main function that you can call/modify to run experiments.
For example:

python sender_generalization.py
or

python recpient_generaliztion.py
(Adjust the scripts as needed for your specific environment/folder names.)

2. GUI Use
For a more user-friendly experience, you can launch the Tkinter GUI:

python combined_gui.py
This GUI has three tabs:

Sender

Click Train Sender Model to run sender.train_model().
Console logs will appear in the text area, and a confusion matrix will be embedded in the tab.
Recipient

Click Train Recipient Model to run recepient.train_model().
Similar logs and confusion matrix will appear.
Test New Data

Select a folder containing subfolders of .wav files, where each subfolder name is the ground-truth label.
The GUI will run predictions with whichever models you trained in tabs 1 or 2, then display confusion matrices and compute test accuracy.
Note: If you have not trained one of the models, the GUI will show a warning or skip that model in the test phase.

Important Details
Random Seeds:
We fix random_state=42 in many scripts for reproducibility. You can change seeds as needed.

Oversampling:
Training data is balanced with BorderlineSMOTE followed by ADASYN, which may slow down training if your dataset is large.

Normalization:
Each .wav file is normalized by dividing by its minimum value (if not zero) before low-pass filtering. This is a simple form of amplitude normalization.

Feature Extraction:
The scripts extract 20 features (time and frequency domain). These include RMS, ZCR, energy, min/max, mean, variance, skewness, kurtosis, spectral entropy, spectral centroid, etc.

Hyperparameter Tuning:
Training uses RandomizedSearchCV over a pre-defined grid of random forest parameters, with 3-fold cross-validation.

Extending or Modifying the Code
If your folder structure or naming scheme differs, modify the parse_sender() or parse_recipient() functions (in sender.py, recepient.py, or the generalization scripts) so that the correct label is extracted from subfolder names.
If you need different hyperparameters, update the param_grid inside train_model() (in either sender.py or recepient.py).
For advanced novelty detection or leaving out certain labels, edit the “generalization” scripts (sender_generalization.py / recpient_generaliztion.py).
License
This project does not have a declared license in this repository. Please include one if you distribute or modify it.

Contact
For questions, bug reports, or collaboration, please open an issue or contact the repository owner.

Thank you for using final_project_spalax!
We hope these tools help you in analyzing and classifying seismic or audio signals. Feel free to adapt the code to fit your specific needs.
