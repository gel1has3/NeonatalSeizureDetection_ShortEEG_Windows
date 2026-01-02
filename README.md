# Neonatal Seizure Detection Short EEG Windows (GST,TW)

It contains a full pipeline, including preprocessing, fine-tuning, training, and evaluation using  **EEGConformer** and **Transformer** models for seizure (`S`) and non-seizure (`NS`) classification.


## Hardware Used

All experiments were conducted on an NVIDIA GPU using PyTorch with CUDA enabled. Training and  evaluation leveraged GPU acceleration for faster computation and model convergence. 
---

## Project Overview

The goal is to classify short windows EEG segments into `S`: Seizure and `NS`: Non-seizure

We use a  workflow that combines:
- **Preprocessing raw EDF EEG data**
- **Segment-wise annotation with seizure and non-seizure tags**
- **Optional relabeling (`D`) of specific intervals (expert annotation disagreement)**
- **Model training using Transformer and EEGConformer**
- **Performance evaluation and visualization**

---

## Dataset Structure

The EEG files and annotations should follow this structure:

```
Neontal\_eeg\_dataset1/
│
├── data/
│   ├── raw_edf/
│   ├── annotations/
│   └── ...
├── models/
│   └── transformer.py
│   └── eegconformer.py
├── train_model.py
├── evaluate_model.py
├── preprocess_pipeline.py
├── update_annotations.py
└── README.md


````

Annotation CSVs:
- `csa_seizures_chunks.csv`: Initial seizure intervals.
- `annotations_difference.csv`: Manually corrected regions to be relabeled (optional).

---

## Preprocessing Pipeline

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
   Each seizure interval is split into 1-second segments and labeled: `S` for seizure and `NS` for non-seizure 

7. **Annotation Saving**  
   Annotated EEG is saved in `.fif` format for downstream use.
8. **Annotation Visualization** 
 ![image](https://github.com/user-attachments/assets/6955639d-9af0-4858-8ee2-3411d049491e)


---

## Annotation Update Workflow (Optional)

Some `.fif` files may need manual annotation correction (e.g., missed or mislabeled events). Use `annotations_difference.csv` to relabel.

Steps:
- Load existing `.fif`
- Remove old annotations that overlap with the specified range
- Add new 1-second segments labeled `D`
- Save the updated file as `eegX_updated.fif`
  
---

## 🧠 Model Architectures

### 1. EEGConformer
- Combines convolutional and Transformer layers.
- Hybrid model combining:
   - CNN for spatial feature extraction
   - Transformer/conformer layers for temporal dynamics
- Excels in capturing both local (spatial) and global (temporal) EEG patterns.

### 2. Transformer 

- Treats each EEG segment as a sequence of features.
- Learns attention-based representations across channels and time.
- Multi-head self-attention model
- Operates on EEG segment sequences
- Lightweight compared to EEGConformer, useful as a strong baseline

## Model Training Strategies

To handle class imbalance (seizure segments are rare), we trained each model using three different strategies:
| Strategy               | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Original**           | Trained on raw distribution (imbalanced).                                     |
| **SMOTE Oversampling** | Synthetic Minority Over-sampling applied on seizure segments to balance data. |
| **Cost-Sensitive**     | Class weights assigned to loss function (`CrossEntropyLoss(weight=...)`).     |

---

## Model Training

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

## Evaluation & Comparison

Models are evaluated using:
| Metric           | Description                                 |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall correct classification rate         |
| Precision (S/NS) | How many predicted seizures were correct    |
| Recall (S/NS)    | How many true seizures were detected        |
| F1-score         | Harmonic mean of precision and recall       |
| Confusion Matrix | For visual inspection of prediction quality |
| **Loss Curves**      | Plots of training and validation loss per epoch        |
| **Accuracy Curves**  | Plots of training and validation accuracy per epoch    |

## Comparison Table (Sample Format)
| Model        | Strategy       | Accuracy  | Precision (S) | Recall (S) | F1-score (S) |
| ------------ | -------------- | --------- | ------------- | ---------- | ------------ |
| Transformer  | Original       | 85.2%     | 0.71          | 0.54       | 0.61         |
| Transformer  | SMOTE          | 88.6%     | 0.76          | 0.68       | 0.72         |
| Transformer  | Cost-Sensitive | 89.1%     | 0.79          | 0.70       | 0.74         |
| EEGConformer | Original       | 87.5%     | 0.74          | 0.66       | 0.70         |
| EEGConformer | SMOTE          | 90.2%     | 0.81          | 0.72       | 0.76         |
| EEGConformer | Cost-Sensitive | **91.0%** | **0.83**      | **0.75**   | **0.79**     |


## Model Comparison Table (Mean ± SD) (Sample Format)
| Model        | Strategy       | Accuracy        | Precision (S)   | Recall (S)      | F1-score (S)    | ROC-AUC         |
| ------------ | -------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Transformer  | Original       | 85.2 ± 1.3%     | 0.71 ± 0.04     | 0.54 ± 0.05     | 0.61 ± 0.04     | 0.82 ± 0.02     |
| Transformer  | SMOTE          | 88.6 ± 1.1%     | 0.76 ± 0.03     | 0.68 ± 0.04     | 0.72 ± 0.03     | 0.86 ± 0.01     |
| Transformer  | Cost-Sensitive | 89.1 ± 0.9%     | 0.79 ± 0.03     | 0.70 ± 0.03     | 0.74 ± 0.02     | 0.88 ± 0.01     |
| EEGConformer | Original       | 87.5 ± 1.2%     | 0.74 ± 0.03     | 0.66 ± 0.03     | 0.70 ± 0.02     | 0.84 ± 0.02     |
| EEGConformer | SMOTE          | 90.2 ± 0.8%     | 0.81 ± 0.02     | 0.72 ± 0.03     | 0.76 ± 0.02     | 0.89 ± 0.01     |
| EEGConformer | Cost-Sensitive | **91.0 ± 0.7%** | **0.83 ± 0.02** | **0.75 ± 0.02** | **0.79 ± 0.02** | **0.91 ± 0.01** |



---

## How to Run

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

## Requirements

* Python 3.8+
* `mne`, `scipy`, `numpy`, `pandas`
* `torch`, `sklearn`, `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```


---

##  Notes

* Ensure that `filters_db_256.mat` is present for filtering step.
* Update `downsample_factor` if original EEGs use different sampling rates.
* Modify EEGConformer input dimensions according to number of channels and sample points after preprocessing.

---


## Acknowledgements

* Dataset based on  "A dataset of neonatal EEG recordings with seizure annotations @https://zenodo.org/records/1280684"


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

