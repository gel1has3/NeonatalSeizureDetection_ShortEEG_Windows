# EEG Annotation Updater based on Expert Disagreement 

This script updates existing EEG `.fif` files with revised annotations provided in a CSV. It is designed for workflows where annotation corrections based on expert disagreement and need to be integrated back into the data for consistency in downstream analyses.

---

## Overview

- **Input**: Existing `.fif` EEG files with annotations and a CSV file containing corrected time intervals.
- **Output**: Updated `.fif` EEG files with revised annotations replacing the outdated ones.
- **Update Type**: Replaces previous annotations with `"D"` (e.g., indicating updated seizure detection).

---

## Workflow

1. **Load EEG file** (`.fif`) with existing annotations.
2. **Identify update windows** in seconds from the correction CSV.
3. **Remove annotations** overlapping with the updated regions.
4. **Insert new annotations** labeled `"D"` for each 1-second segment within the update window.
5. **Save** the EEG file with the new annotations.

---

## File Structure

```plaintext
📁 eeg_annotation_updater/
├── eeg_folder/               # Directory with original .fif EEG files
│   ├── eeg1.fif
│   └── eeg2.fif
├── annotations_difference.csv # CSV with updated annotation intervals
├── updated_annotations/      # Output directory for updated .fif files
└── update_annotations.py     # Main script
