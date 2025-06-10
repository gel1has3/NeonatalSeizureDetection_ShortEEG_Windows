# EEG Epoch Creation and Dataset Preparation for ML/DL/Transformer
This repository provides a set of utility functions to load, filter, and prepare EEG epoch data from `.fif` (MNE) or `.set` (EEGLAB) files. These functions are ideal for preprocessing EEG data for downstream tasks like classification, deep learning, or feature extraction.

---

## Features

- Load EEG epochs from `.fif` or `.set` files.
- Filter epochs by annotation/event labels.
- Prepare a combined dataset from an entire folder of epoch files.
- Returns data in NumPy array format ready for machine learning.

---

## Function Overviews

### `load_epochs(file_path: str, format: str = 'fif') -> mne.Epochs`

**Description:**  
Loads epochs from a `.fif` or `.set` file using MNE.

**Parameters:**
- `file_path` (`str`): Path to the epoch file.
- `format` (`str`): File format (`'fif'` or `'set'`). Default is `'fif'`.

**Returns:**  
- `mne.Epochs` object

---

### `filter_epochs(epochs: mne.Epochs, keep_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]`

**Description:**  
Filters the loaded epochs based on specified annotation labels and extracts the data and corresponding labels.

**Parameters:**
- `epochs` (`mne.Epochs`): Epoch object loaded from file.
- `keep_labels` (`List[str]`): List of label names to keep.

**Returns:**  
- `data`: `np.ndarray` of shape `(n_epochs, n_channels, n_times)`
- `labels`: `np.ndarray` of shape `(n_epochs,)`

---

### `prepare_dataset_from_folder(folder_path: str, keep_labels: List[str], format: str = 'fif') -> Tuple[np.ndarray, np.ndarray]`

**Description:**  
Loads all epoch files in a folder, filters them by label, and aggregates the data into a single dataset.

**Parameters:**
- `folder_path` (`str`): Path to the directory containing epoch files.
- `keep_labels` (`List[str]`): Labels to keep for filtering.
- `format` (`str`): Format of files to load (`'fif'` or `'set'`). Default is `'fif'`.

**Returns:**
- `X`: `np.ndarray` of EEG data with shape `(total_epochs, n_channels, n_times)`
- `y`: `np.ndarray` of integer labels with shape `(total_epochs,)`

---

## Example Usage

You can either open it in Jupyter Notebook 
or 

```python
from dataset_utils import prepare_dataset_from_folder  # Assuming functions are in dataset_utils.py

# Define folder and labels of interest
folder = "path/to/epoch_files"
labels_to_keep = ['seizure', 'non-seizure']

# Load and prepare dataset
X, y = prepare_dataset_from_folder(folder, keep_labels=labels_to_keep, format='fif')

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
