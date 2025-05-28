# NeonatalSeizureDetection_ShortEEG_Windows
- **Geletaw Sahle Tegenaw, Ph.D**
- **Norman Delanty, Prof.**
- **Tomas Ward, Prof.**

This repository contains the full pipeline, including preprocessing, fine-tuning, training, and evaluation using **EEGConformer** and **Transformer** models for seizure (`S`) and non-seizure (`NS`) classification.

---

## 📋 Project Overview

The goal is to classify short windows EEG segments into
- `S`: Seizure
- `NS`: Non-seizure

We use a  workflow that combines:
- **Preprocessing raw EDF EEG data**
- **Segment-wise annotation with seizure and non-seizure tags**
- **Optional relabeling (`D`) of specific intervals**
- **Model training using Transformer and EEGConformer**
- **Performance evaluation and visualization**

---

## 📂 Dataset Structure

The EEG files and annotations should follow this structure:

```

Neontal\_eeg\_dataset1/
│
├── eeg1.edf
├── eeg2.edf
├── ...
└── annotations/
├── eeg1.fif
├── eeg2.fif
└── ...

````

Annotation CSVs:
- `csa_seizures_chunks.csv`: Initial seizure intervals.
- `annotations_difference.csv`: Manually corrected regions to be relabeled (optional).

---

## 🧠 Preprocessing Pipeline

All EEG files go through the following steps:

1. **Load & Validate EDF**  
   Ensures file exists and is in correct format.

2. **Rename Channels**  
   Converts labels like `EEG FP1-REF` → `FP1`.

3. **Bipolar Montage Construction**  
   Computes differences between standard 10-20 electrode pairs.

4. **Downsampling**  
   Reduces sample rate from 256 Hz → 64 Hz (by a factor of 4).

5. **Filtering**  
   Applies:
   - Bandpass filter (0.3–30 Hz)
   - 50 Hz notch filter
   - Custom filters from `.mat` files

6. **Annotation Assignment**  
   Each seizure interval is split into 1-second segments and labeled:
   - `S` for seizure
   - `NS` for non-seizure (complement region)

7. **Annotation Saving**  
   Annotated EEG is saved in `.fif` format for downstream use.
   
9. **Annotation Visualization **  
   ![image](https://github.com/user-attachments/assets/40971b57-4701-401c-8cd9-81dea935fff0)

---

## 🔁 Annotation Update Workflow (Optional)

Some `.fif` files may need manual annotation correction (e.g., missed or mislabeled events). Use `annotations_difference.csv` to relabel .

Steps:
- Load existing `.fif`
- Remove old annotations that overlap with the specified range
- Add new 1-second segments labeled `D`
- Save the updated file as `eegX_updated.fif`

---

## 🧠 Model Architectures

### 1. EEGConformer

- Combines convolutional and Transformer layers.
- Excels in capturing both local (spatial) and global (temporal) EEG patterns.

### 2. Transformer (Vanilla)

- Treats each EEG segment as a sequence of features.
- Learns attention-based representations across channels and time.

---

## 🏋️ Model Training

### Input
- Annotated EEG segments (1 sec each) with labels: `S`, `NS`
- Data is reshaped into `channels × time` matrices

### Strategy
- Supervised classification with CrossEntropyLoss
- Balanced batches to avoid label imbalance

### Tools
- PyTorch / PyTorch Lightning
- Scikit-learn for metric evaluation
- WandB or TensorBoard for monitoring (optional)

---

## 📊 Evaluation

Models are evaluated using:

- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**
- **ROC-AUC** (if applicable)

Optional:
- Attention visualization (for interpretability)
- Per-patient performance breakdown

---

## 🔧 How to Run

### 1. Preprocess & Annotate

```bash
python preprocess_pipeline.py
````

### 2. Update Annotations (Optional)

```bash
python update_annotations.py --corrections annotations_difference.csv
```

### 3. Train Model

```bash
python train_model.py --model transformer --epochs 50
```

### 4. Evaluate

```bash
python evaluate_model.py --model-path saved_models/transformer_final.pt
```

---

## 📎 Requirements

* Python 3.8+
* `mne`, `scipy`, `numpy`, `pandas`
* `torch`, `sklearn`, `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* Ensure that `filters_db_256.mat` is present for filtering step.
* Update `downsample_factor` if original EEGs use different sampling rates.
* Modify EEGConformer input dimensions according to number of channels and sample points after preprocessing.

---

## 👩‍🔬 Acknowledgements

* Dataset based on  "A dataset of neonatal EEG recordings with seizure annotations @https://zenodo.org/records/1280684"





