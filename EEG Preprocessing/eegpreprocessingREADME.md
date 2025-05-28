`README.md` file for your EEG preprocessing and annotation pipeline using `.edf` files

---


# Neonatal EEG Preprocessing  Pipeline

## Overview

**NeuroInsight** is a robust Python-based pipeline for preprocessing neonatal EEG recordings for seizure analysis. This includes signal standardization, bipolar montage generation, advanced filtering, segmentation, and labeling based on expert annotations. The goal is to provide clean and labeled EEG data suitable for downstream machine learning tasks and clinical research. This repository provides a complete EEG processing pipeline for neonatal seizure detection using `.edf` EEG recordings. The pipeline includes:

- Loading and validating `.edf` files
- Standardizing EEG channel names
- Creating a bipolar montage
- Downsampling and filtering signals
- Annotating seizure (`S`) and non-seizure (`NS`) segments based on CSV annotation input
- Saving the annotated EEG files in `.fif` format for further analysis or visualization

---

## Folder Structure

```plaintext
📁 Neontal_eeg_dataset1/
├── eeg1.edf                      # Original EDF EEG file
├── eeg2.edf                      # Additional EEG recordings
├── annotations/                 # Output directory for annotated .fif files
│   └── eeg1.fif
📄 EEG_ML.ipynb                   # Notebook executing the pipeline
📄 csa_seizures_chunks.csv        # CSV file containing seizure annotation time ranges
📄 filters_db_256.mat             # MATLAB file containing custom filter coefficients
````

---

## Requirements

Install all required dependencies:

```bash
pip install mne pandas numpy scipy
```

---

## Core Functions

### `load_edf_file(file_path)`

Loads an `.edf` EEG file with robust error handling.

### `rename_eeg_channels(raw)`

Standardizes channel names by removing prefixes/suffixes like `"EEG "` and `"-REF"`.

### `create_bipolar_montage(raw)`

Generates a bipolar montage using the 10–20 system (e.g., `Fp1-F3`, `F3-C3`).

### `downsample_eeg(raw_bipolar, downsample_factor=4)`

Reduces the sampling frequency (e.g., from 256 Hz to 64 Hz).

### `get_filter_coefficients(raw)`

Applies a series of filters (notch, bandpass) using predefined MATLAB `.mat` filter coefficients.

---

## EEG Annotation Workflow

1. **Define File Paths**

   ```python
   eeg_folder = "Neontal_eeg_dataset1"
   annotation_folder = os.path.join(eeg_folder, "annotations")
   ```

2. **Load Annotation CSV**

   The CSV file (`csa_seizures_chunks.csv`) should include `Id`, `from_sec`, `to_sec` columns specifying seizure intervals.

3. **Loop Through EDF Files**

   For each EEG file:

   * Load and preprocess the raw EEG signal
   * Annotate seizure (`S`) segments based on CSV intervals
   * Automatically fill in the remaining segments as non-seizure (`NS`)
   * Save as `.fif` file with all annotations

4. **Resulting Output**

   Annotated EEG files are saved under:

   ```
   Neontal_eeg_dataset1/annotations/eegX.fif
   ```

---

## Example Code Snippet

You can either open it in Jupyter Notebook

or 


```python
raw = mne.io.read_raw_fif("Neontal_eeg_dataset1/annotations/eeg1.fif", preload=True)
raw.plot(start=100, duration=20)
```

This will visualize a 20-second window of the annotated EEG starting at the 100-second mark. like
![image](https://github.com/user-attachments/assets/adc6cb5e-15f5-4a40-b194-0a614a4cd91a)

---

## Final Output

* Annotated EEGs with consistent channel naming and bipolar montages
* Each file includes seizure and non-seizure events in 1-second resolution
* Output is in `.fif` format, ready for  machine learning and visualization

---

## License

This code is provided for academic and research use. Licensing for clinical or commercial use may require permission.

