This script updates EEG `.fif` files with new seizure annotations based on a CSV containing differences based on expert disagreements.

---

````markdown
# EEG Annotation Updater

This script updates existing EEG `.fif` files with revised annotations provided in a CSV. It is designed for workflows where annotation corrections or refinements are applied post hoc and need to be integrated back into the data for consistency in downstream analyses.

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
````

---

## CSV Format (`annotations_difference.csv`)

The CSV must contain:

| Id | from\_sec | to\_sec |
| -- | --------- | ------- |
| 1  | 45        | 50      |
| 2  | 100       | 110     |

* `Id`: Numeric part of the EEG filename (e.g., `eeg1.fif` → `1`)
* `from_sec` and `to_sec`: Time range in seconds to update annotations.

---

## Requirements

Install the necessary Python libraries:

```bash
pip install mne pandas numpy
```

---

## Usage

Update the script paths, then run:

```python
import os
import mne
import pandas as pd

# Set paths
eeg_folder = '/path/to/eeg_folder'
annotation_folder = '/path/to/updated_annotations'
annotations_difference_df = pd.read_csv('/path/to/annotations_difference.csv')

# Loop through and update EEG files
eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith('.fif')]
for eeg_file in eeg_files:
    file_id = int(eeg_file.replace("eeg", "").replace(".fif", ""))
    
    updated_annotations = annotations_difference_df[annotations_difference_df['Id'] == file_id]
    if updated_annotations.empty:
        print(f"No updated annotations for {eeg_file}. Skipping update...")
        continue

    raw = mne.io.read_raw_fif(os.path.join(eeg_folder, eeg_file), preload=True)
    ann_df = pd.DataFrame({
        "onset": raw.annotations.onset,
        "duration": raw.annotations.duration,
        "description": raw.annotations.description
    })

    new_onsets, new_durations, new_descriptions = [], [], []
    for _, row in updated_annotations.iterrows():
        update_range = set(range(int(row['from_sec']), int(row['to_sec'])))
        ann_df = ann_df[~ann_df.apply(
            lambda x: any(int(x.onset + i) in update_range for i in range(int(x.duration))),
            axis=1
        )]
        for t in update_range:
            new_onsets.append(t)
            new_durations.append(1)
            new_descriptions.append("D")

    updated_ann_df = pd.concat([
        ann_df,
        pd.DataFrame({
            "onset": new_onsets,
            "duration": new_durations,
            "description": new_descriptions
        })
    ]).sort_values(by="onset").reset_index(drop=True)

    raw.set_annotations(mne.Annotations(
        onset=updated_ann_df["onset"].values,
        duration=updated_ann_df["duration"].values,
        description=updated_ann_df["description"].values
    ))

    updated_path = os.path.join(annotation_folder, eeg_file.replace(".edf", "_updated.fif"))
    raw.save(updated_path, overwrite=True)

    print(f"Updated annotations saved to: {updated_path}")
```

---

## Output

* **Format**: `.fif` files with updated MNE annotations
* **Directory**: `updated_annotations/`
* **Annotation Label**: `"D"` for updated intervals

---

## License

This code is free to use for academic research purposes. Attribution to the authors of the original EEG dataset and preprocessing pipeline is appreciated.

