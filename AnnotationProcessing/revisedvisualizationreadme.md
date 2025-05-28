## Updating EEG Annotations with Revised Segments

### Purpose

This step refines the previously annotated `.fif` EEG files by incorporating revised or corrected annotation segments, particularly regions that need to be relabeled as `"D"` ( denoting difference of annonations).

This process ensures that:
- New annotations are inserted based on updated time intervals provided in a separate CSV.

---

### Input Requirements

- **Preprocessed `.fif` EEG files**: Annotated EEGs from the initial pipeline.
- **Updated annotations CSV (`annotations_difference_df`)**: A DataFrame (or CSV) with columns:
  - `Id`: File identifier (extracted from filename like `eeg1.fif`)
  - `from_sec`, `to_sec`: Time intervals (in seconds) needing new annotations.

---

### Workflow Summary

1. **Iterate Over EEG Files**  
   For each `.fif` file, extract its numeric ID and check if there are any updated annotation segments.

2. **Load Existing Annotations**  
   - Retrieve and convert the current annotations to a DataFrame for flexible filtering.
   - Remove all 1-second annotation segments that overlap with the new `"from_sec"` to `"to_sec"` ranges.

3. **Insert Updated Annotations (`"D"`)**  
   - Add new 1-second segments labeled as `"D"` for the specified time ranges.

4. **Merge and Sort**  
   - Combine the filtered existing annotations with the new `"D"` segments.
   - Sort chronologically by onset time.

5. **Save Updated File**  
   - Save the updated annotations into a new `.fif` file (with `_updated.fif` suffix) in the annotations folder.

---

### Output
Each EEG file with updated annotations will be saved as:

Neontal_eeg_dataset1/annotations/eegX_updated.fif

These can now be visualized or used for model training with corrected labels as

![d](https://github.com/user-attachments/assets/f2069073-88b5-4dea-a0e8-133459592354)

---
### Notes
This process preserves non-overlapping original annotations.

The "D" label can be reinterpreted depending on your project's specific labeling schema (e.g., "Differences")

---
### Code Snippet

```python
for eeg_file in eeg_files:
    file_id = int(eeg_file.replace("eeg", "").replace(".fif", ""))
    updated_annotations = annotations_difference_df[annotations_difference_df['Id'] == file_id]
    if updated_annotations.empty:
        continue

    raw = mne.io.read_raw_fif(os.path.join(eeg_folder, eeg_file), preload=True)
    ann_df = pd.DataFrame({
        "onset": raw.annotations.onset,
        "duration": raw.annotations.duration,
        "description": raw.annotations.description
    })

    new_onsets, new_durations, new_descriptions = [], [], []
    for _, row in updated_annotations.iterrows():
        from_sec, to_sec = row['from_sec'], row['to_sec']
        update_range = set(range(int(from_sec), int(to_sec)))

        # Remove overlapping existing annotations
        ann_df = ann_df[~ann_df.apply(
            lambda x: any(int(x.onset + i) in update_range for i in range(int(x.duration))),
            axis=1
        )]

        # Add new "D" annotations
        for t in update_range:
            new_onsets.append(t)
            new_durations.append(1)
            new_descriptions.append("D")

    # Combine and sort
    updated_ann_df = pd.concat([
        ann_df,
        pd.DataFrame({
            "onset": new_onsets,
            "duration": new_durations,
            "description": new_descriptions
        })
    ]).sort_values(by="onset").reset_index(drop=True)

    # Apply to raw and save
    raw.set_annotations(mne.Annotations(
        onset=updated_ann_df["onset"],
        duration=updated_ann_df["duration"],
        description=updated_ann_df["description"]
    ))
    raw.save(os.path.join(annotation_folder, eeg_file.replace(".edf", "_updated.fif")), overwrite=True)




