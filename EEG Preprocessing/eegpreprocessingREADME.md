Here's a comprehensive `README.md` file tailored for the **NeuroInsight: Neonatal EEG Preprocessing** project. This file documents the dataset, preprocessing goals, methodology, and usage details for users or collaborators.

---

# 🧠 NeuroInsight: Neonatal EEG Preprocessing

**Authors**:
Geletaw Sahle Tegenaw (Ph.D), Norman Delanty (Prof), Tomas Ward (Prof)

## Overview

NeuroInsight is a neonatal EEG preprocessing framework built to facilitate automated neonatal seizure detection by applying robust preprocessing pipelines to raw EEG recordings. The foundation of this work lies in detecting the periodic and evolving patterns indicative of seizures in neonates.

---

## 📋 Neonatal EEG Dataset Summary

* **Source**: NICU, Helsinki University Hospital
* **Recording Period**: 2010–2014
* **Total Subjects**: 79 neonates
* **Data Format**:

  * 79 EDF (European Data Format) raw EEG files
  * Annotation files: 3 expert annotations (CSV, MAT formats)
* **Electrode Setup**:

  * 19 electrodes (10-20 system)
  * Bipolar montage: 18 channel pairs
* **Sampling Rate**: 256 Hz
* **Annotation**:

  * Provided by 3 independent human experts (A, B, C)
  * Annotations sampled at 1-second resolution
* **Seizure Annotations**:

  * Total annotated seizures: **1,379**
  * 40 neonates with consistent expert annotations
  * 22 seizure-free neonates
  * 17 neonates excluded due to partial annotations
* **Dataset Link**: [Zenodo - 4940267](https://zenodo.org/record/4940267)

---

## ⚙️ Key Preprocessing Pipeline

### ✔️ File Loading

* Reads `.edf` EEG files using MNE with error handling.

### ✔️ Channel Name Standardization

* Cleans and renames channels to remove prefixes/suffixes like `EEG` and `-REF`.

### ✔️ Bipolar Montage Creation

* Constructs 18-channel bipolar montage from standard 10-20 pairs.

### ✔️ Filtering

* **Filters Applied**:

  * Custom filters from provided `.mat` file (`filters_db_256.mat`)
  * 50 Hz notch filter
  * 0.3–30 Hz bandpass Butterworth filter
  * Zero-phase filtering via `filtfilt` to avoid distortion

### ✔️ Downsampling

* EEG is downsampled from 256 Hz to **64 Hz** (factor of 4).

### ✔️ Segmentation and Labeling

* EEG is split into 1-second segments.
* Each segment labeled as `seizure` or `noseizure` using expert annotations.

### ✔️ HDF5 Output Format

* EEG segments saved as `.h5` files for efficiency.
* File naming includes expert, ID, label, and time segment.

---

## 📁 Directory Structure

```bash
project/
│
├── data/
│   ├── C1_seizure/       # Segments labeled as seizures
│   └── C1_noseizure/     # Segments labeled as non-seizures
│
├── filters_db_256.mat    # Filter coefficients
├── seizures_chunks.csv   # Annotations from experts
└── preprocessing_script.py
```

---

## Usage Guide

### 💾 Paths to Modify

Ensure the following paths are correctly set in your script:

```python
edfdirectly_path = '/path/to/edf/files'
annotation_path = '/path/to/seizures_chunks.csv'
```

### Preprocessing Execution

```python
process_edf_files(edf_directory=edfdirectly_path, 
                  csv_annotations_path=annotation_path, 
                  experts=['A', 'B', 'C'])
```

Each EDF file will be:

* Renamed
* Converted to bipolar montage
* Filtered
* Downsampled
* Segmented
* Saved with appropriate labels

---

## Segment Classification Logic

A 1-second EEG segment is labeled `seizure` **if it falls entirely** within an expert-annotated seizure interval; otherwise, it's labeled `noseizure`.

```python
def get_segment_label(start_time, end_time, intervals):
    for _, row in intervals.iterrows():
        if start_time >= row['from_sec'] and end_time <= row['to_sec']:
            return 'seizure'
    return 'noseizure'
```

---

##  Dependencies

* Python 3.7+
* Libraries:

  * `mne`, `numpy`, `pandas`, `matplotlib`, `scipy`, `h5py`

Install with:

```bash
pip install mne numpy pandas matplotlib scipy h5py
```

---

##  EEG Subject Groups

```python
# Seizure-annotated by all 3 experts
sIDs = [1, 4, 5, 7, 9, 11, ..., 79]

# Seizure-free
nsIDs = [3, 10, 18, ..., 72]
```

---

## Citation

If you use this code or dataset, please cite:

> [Zenodo Dataset - DOI:10.5281/zenodo.4940267](https://zenodo.org/record/4940267)

---

## Contact

For questions or contributions, please contact **Geletaw Sahle Tegenaw**
Email: *\[[your-email@example.com](mailto:your-email@example.com)]*
GitHub: \[your-github-link]

---

Let me know if you’d like the README saved to a file, enhanced with badges or links, or transformed into a Jupyter-compatible Markdown cell.

