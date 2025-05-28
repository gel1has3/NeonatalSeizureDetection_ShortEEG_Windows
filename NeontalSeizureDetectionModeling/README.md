
---

# EEG Neonatal Seizure Detection with EEGConformer

EEG training and evaluation pipeline using the EEGConformer model, with complete handling of **data imbalance**, **SMOTE**, and **cost-sensitive learning**, along with training, evaluation, and visualization. This repository implements a  pipeline for classifying EEG segments into **Seizure (S)** and **Non-Seizure (NS)** using the **EEGConformer** model. It supports training with:

* Standard supervised learning
* **Data imbalance mitigation** using SMOTE
* **Cost-sensitive learning** using weighted cross-entropy loss

The codebase includes model training, evaluation, and visualization routines such as ROC curves, confusion matrices, and learning curves.

---

##  Project Structure

```
.
├── data/
│   └── preprocessed_eeg.npy  # (Optional) Store preprocessed EEG arrays
├── model/
│   └── EEGConformer()        # Model defined using Braindecode
├── utils/
│   ├── plots.py              # Visualization functions
│   └── metrics.py            # Metric computations
├── train.py                  # End-to-end training and evaluation script
├── README.md                 # This file
```

---

## Requirements

Install dependencies:

```bash
pip install torch numpy matplotlib seaborn scikit-learn imbalanced-learn braindecode
```

---

## Training Pipeline Overview

### 1. Dataset Preparation

We use a custom `EEGDataset` class to wrap EEG feature arrays (`X`) and corresponding labels (`y`).

```python
dataset = EEGDataset(X, y)
```

Split the dataset into training and validation sets:

```python
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, val_idx)
```

Optionally, apply **SMOTE** to the training set for class balancing:

```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X[train_idx], y[train_idx])
train_dataset = EEGDataset(X_resampled, y_resampled)
```

---

### 2. Model: EEGConformer

```python
from braindecode.models import EEGConformer

model = EEGConformer(
    n_outputs=2,
    n_chans=18,
    n_times=64,
    n_filters_time=40,
    filter_time_length=16,
    pool_time_length=8,
    pool_time_stride=4,
    att_depth=6,
    att_heads=10,
    drop_prob=0.5,
    final_fc_length="auto"
).to(device)
```

---

### 3. Loss Functions

* **Default**: `nn.CrossEntropyLoss()`
* **Cost-sensitive**: Compute class weights inversely proportional to frequency

```python
weights = torch.tensor([1.0, class_weight_ratio], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

### 4. Training

Use the `train_model()` function to train and validate the model:

```python
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
```

It returns:

* Training/validation loss and accuracy per epoch
* Predicted probabilities and labels for ROC, PR curve, and confusion matrix

---

##  Evaluation Metrics & Plots

| Metric           | Description                                 |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall classification accuracy             |
| Precision        | Positive prediction correctness (S class)   |
| Recall           | Sensitivity or true positive rate (S class) |
| F1-score         | Harmonic mean of precision and recall       |
| ROC-AUC          | Area under ROC curve                        |
| PR Curve         | Precision-Recall curve                      |
| Confusion Matrix | Count of TP, FP, TN, FN                     |
| Loss Curves      | Plots of training and validation loss per epoch        |
| Accuracy Curves  | Plots of training and validation accuracy per epoch    |

### Visualizations

All plots are automatically saved to disk:

* `EEGConformerTrainingandValidationLoss.png`
* `EEGConformerTrainingandValidationAccuracy.png`
* `EEGConformer_ROC.png`
* `EEGConformer_Precision-RecallCurve.png`
* `CustomConfusionMatrix.png`

---

##  Comparison of Learning Strategies

| Strategy       | Handling Imbalance    | Loss Function         | SMOTE Applied | Class Weights |
| -------------- | --------------------- | --------------------- | ------------- | ------------- |
| Baseline       | ❌                     | CrossEntropy          | ❌             | ❌             |
| SMOTE          | ✅ (oversampling)      | CrossEntropy          | ✅             | ❌             |
| Cost-sensitive | ✅ (penalize minority) | Weighted CrossEntropy | ❌             | ✅             |

---

## 📁 Output Files

| File                          | Description             |
| ----------------------------- | ----------------------- |
| `*.png`                       | Evaluation plots        |
| `train_model()` return values | NumPy arrays of metrics |
| `EEGConformer.pth` (if saved) | Trained model weights   |

---

##  Example Results (on validation set)

```text
Epoch [100/100] Train Loss: 0.2331, Train Acc: 0.9032 | Val Loss: 0.2954, Val Acc: 0.8821

Model Metrics:
Accuracy: 0.8821
Precision: 0.8456
Recall: 0.7983
F1-Score: 0.8213

Classification Report:
              precision    recall  f1-score   support
   Negative       0.92      0.94      0.93       200
   Positive       0.85      0.80      0.82       100
```

---

## Reproducibility

Set seeds for full reproducibility across Torch, NumPy, and CUDA:

```python
set_seed(42)
```

---

## To Run

```bash
python train.py --strategy smote --epochs 100 --lr 0.001 --batch_size 64
```

---

