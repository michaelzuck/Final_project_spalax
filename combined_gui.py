import os
import io
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import your existing modules (ensure these files are in the same folder or on the Python path)
import sender
import recipient


###############################################################################
# Helper: override plt.show() to capture the last figure
###############################################################################
class FigureCapture:
    """
    Temporarily override plt.show() to store the last figure created.
    This lets us embed that figure in the Tkinter GUI instead of popping up a new window.
    """
    def __init__(self):
        self.old_show = plt.show
        self.captured_figure = None

    def replace(self):
        plt.show = self._my_show

    def restore(self):
        plt.show = self.old_show

    def _my_show(self, *args, **kwargs):
        self.captured_figure = plt.gcf()
        # We skip the actual UI display so it can be manually embedded in Tk.


###############################################################################
# Helper: parse final accuracy from logs
###############################################################################
def parse_accuracy_from_logs(logs_text):
    """
    Searches the logs for a line containing 'Accuracy on Test Set:' or similar
    and extracts the numeric value. Returns the *last* found float or None.
    """
    import re
    pattern = re.compile(r"Accuracy on Test Set:\s*(\d*\.\d+|\d+)", re.IGNORECASE)

    lines = logs_text.strip().split("\n")
    final_acc = None
    for line in lines:
        match = pattern.search(line)
        if match:
            final_acc = float(match.group(1))
    return final_acc


###############################################################################
# Helper: Load labeled .wav data from subfolders for "Test New Data" tab
###############################################################################
def load_test_data(root_directory, parse_label):
    """
    Recursively walks `root_directory`. Each subfolder name is the 'true label'.
    Collect all .wav files, parse the label from subfolder name, return (X, y, files).
    Uses the same signal processing as in sender.py to remain consistent.
    """
    from scipy.io import wavfile
    from sender import process_signal, extract_features  # Reuse from your sender code

    X_data = []
    y_data = []
    filepaths = []
    cutoff = 500.0

    for root, dirs, files in os.walk(root_directory):
        rel = os.path.relpath(root, root_directory)
        if rel == ".":
            continue
        true_label = parse_label(rel)  # e.g., subfolder name => label

        for f in files:
            if f.lower().endswith(".wav"):
                fp = os.path.join(root, f)
                try:
                    sr, data = wavfile.read(fp)
                    if len(data.shape) == 2:
                        data = data.mean(axis=1)
                    mn = np.min(data)
                    if mn != 0:
                        data = data / mn
                    filtered = process_signal(data, sr, cutoff)
                    feats = extract_features(filtered, sr)
                    X_data.append(feats)
                    y_data.append(true_label)
                    filepaths.append(fp)
                except Exception as e:
                    print(f"[ERROR reading {fp}]: {e}")

    return np.array(X_data), np.array(y_data), filepaths


###############################################################################
# Main GUI
###############################################################################
class CombinedGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Spalax & Seismic Wave – Classification GUI")
        self.master.geometry("1000x700")

        # Attempt using 'aqua' theme (Mac) or fallback if not available
        self.style = ttk.Style()
        try:
            self.style.theme_use("aqua")
        except:
            pass
        # Increase font sizes
        self.style.configure(".", font=("Helvetica", 14), foreground="#000000")
        self.style.configure("TButton", font=("Helvetica", 14, "bold"))

        # Notebook with 3 tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Sender
        self.tab_sender = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_sender, text="Sender")
        self._build_tab_sender()

        # Tab 2: Recipient
        self.tab_recipient = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_recipient, text="Recipient")
        self._build_tab_recipient()

        # Tab 3: Test New Data
        self.tab_test = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_test, text="Test New Data")
        self._build_tab_test()

        # Model references
        self.sender_model = None
        self.recipient_model = None

    ############################################################################
    # TAB 1: Sender
    ############################################################################
    def _build_tab_sender(self):
        frm = ttk.Frame(self.tab_sender)
        frm.pack(fill=tk.BOTH, expand=True)

        # Train button
        btn = ttk.Button(frm, text="Train Sender Model", command=self.train_sender)
        btn.pack(anchor=tk.W, pady=5)

        # Text area for logs
        self.sender_log = scrolledtext.ScrolledText(frm, wrap=tk.WORD, height=14)
        self.sender_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # A placeholder to show the confusion matrix figure after training
        self.sender_cm_canvas = None

    def train_sender(self):
        """
        Captures all console output from sender.train_model() (which calls plt.show())
        and places the confusion matrix inside this tab.
        """
        self.sender_log.delete("1.0", tk.END)  # Clear the log

        # Capture figure from any plt.show() calls
        capture = FigureCapture()
        capture.replace()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                trained_model, X_train_global, y_train_global = sender.train_model()
                self.sender_model = trained_model
            except Exception as e:
                print("[ERROR in sender train]:", e)

        capture.restore()

        # Show logs
        logs = buffer.getvalue()
        self.sender_log.insert(tk.END, logs)

        # If a confusion matrix figure was produced, embed it in the tab
        if capture.captured_figure:
            if self.sender_cm_canvas:
                self.sender_cm_canvas.get_tk_widget().destroy()

            fig = capture.captured_figure
            canvas = FigureCanvasTkAgg(fig, master=self.tab_sender)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            self.sender_cm_canvas = canvas

    ############################################################################
    # TAB 2: Recipient
    ############################################################################
    def _build_tab_recipient(self):
        frm = ttk.Frame(self.tab_recipient)
        frm.pack(fill=tk.BOTH, expand=True)

        # Train button
        btn = ttk.Button(frm, text="Train Recipient Model", command=self.train_recipient)
        btn.pack(anchor=tk.W, pady=5)

        # Text area for logs
        self.recipient_log = scrolledtext.ScrolledText(frm, wrap=tk.WORD, height=14)
        self.recipient_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # A placeholder for confusion matrix figure
        self.recipient_cm_canvas = None

    def train_recipient(self):
        """
        Captures console output from recipient.train_model(), including any plt.show() calls.
        Embeds confusion matrix figure after training.
        """
        self.recipient_log.delete("1.0", tk.END)

        capture = FigureCapture()
        capture.replace()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                trained_model, X_train_global, y_train_global = recipient.train_model()
                self.recipient_model = trained_model
            except Exception as e:
                print("[ERROR in recipient train]:", e)

        capture.restore()

        logs = buffer.getvalue()
        self.recipient_log.insert(tk.END, logs)

        if capture.captured_figure:
            if self.recipient_cm_canvas:
                self.recipient_cm_canvas.get_tk_widget().destroy()

            fig = capture.captured_figure
            canvas = FigureCanvasTkAgg(fig, master=self.tab_recipient)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            self.recipient_cm_canvas = canvas

    ############################################################################
    # TAB 3: Test New Data
    ############################################################################
    def _build_tab_test(self):
        frm = ttk.Frame(self.tab_test)
        frm.pack(fill=tk.BOTH, expand=True)

        lab = ttk.Label(frm, text="Select a folder with subfolders named by the true label:")
        lab.pack(anchor=tk.W, pady=5)

        btn_browse = ttk.Button(frm, text="Browse & Test", command=self.test_new_data)
        btn_browse.pack(anchor=tk.W, pady=5)

        # Frame to hold the results
        self.results_frame = ttk.Frame(frm)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # We'll have two accuracy labels and two confusion matrix canvases here
        self.sender_acc_label = ttk.Label(self.results_frame, text="Sender Accuracy: N/A")
        self.sender_acc_label.pack(anchor=tk.W, pady=5)
        self.sender_canvas = None

        self.recipient_acc_label = ttk.Label(self.results_frame, text="Recipient Accuracy: N/A")
        self.recipient_acc_label.pack(anchor=tk.W, pady=5)
        self.recipient_canvas = None

    def test_new_data(self):
        """
        Allows the user to select a folder with subfolders named by the ground-truth label.
        Then each .wav file is processed, and we compute confusion matrices and accuracies
        for both the Sender model and the Recipient model, if available.
        """
        folder = filedialog.askdirectory(title="Select Test Data Root Folder")
        if not folder:
            return

        # Check if models are trained
        if (not self.sender_model) and (not self.recipient_model):
            messagebox.showwarning("No Models", "Please train both models before testing.")
            return

        # Load data
        parse_label = lambda x: x  # subfolder name is the label
        X_data, y_data, filepaths = load_test_data(folder, parse_label)

        if len(X_data) == 0:
            messagebox.showinfo("No Data", "No WAV files found in subfolders.")
            return

        # Evaluate Sender
        if self.sender_model:
            self.show_confusion_matrix_and_accuracy(self.sender_model, X_data, y_data, is_sender=True)
        else:
            self.sender_acc_label.config(text="Sender Accuracy: Model not available")

        # Evaluate Recipient
        if self.recipient_model:
            self.show_confusion_matrix_and_accuracy(self.recipient_model, X_data, y_data, is_sender=False)
        else:
            self.recipient_acc_label.config(text="Recipient Accuracy: Model not available")

    def show_confusion_matrix_and_accuracy(self, model, X, y_true, is_sender=True):
        """
        Generates a confusion matrix and accuracy for the given model vs. X_data, y_true.
        Embeds the figure in the 'Test New Data' tab.
        """
        # Encode labels -> integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_true)

        # Model predictions
        y_pred = model.predict(X)
        # If predictions are strings, encode them so we can compute accuracy
        if y_pred.dtype.kind in ('U', 'S', 'O'):
            y_pred_encoded = le.transform(y_pred)
        else:
            y_pred_encoded = y_pred

        acc = accuracy_score(y_encoded, y_pred_encoded)
        cm = confusion_matrix(y_encoded, y_pred_encoded)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(5,4), dpi=100)
        disp.plot(ax=ax, cmap=plt.cm.Blues)

        if is_sender:
            ax.set_title("Sender Confusion Matrix")
        else:
            ax.set_title("Recipient Confusion Matrix")

        # Replace old canvas if any
        if is_sender and self.sender_canvas:
            self.sender_canvas.get_tk_widget().destroy()
        if (not is_sender) and self.recipient_canvas:
            self.recipient_canvas.get_tk_widget().destroy()

        # Embed in the results frame
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        if is_sender:
            self.sender_acc_label.config(text=f"Sender Accuracy: {acc * 100:.2f}%")
            self.sender_canvas = canvas
        else:
            self.recipient_acc_label.config(text=f"Recipient Accuracy: {acc * 100:.2f}%")
            self.recipient_canvas = canvas


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedGUI(root)
    root.mainloop()
