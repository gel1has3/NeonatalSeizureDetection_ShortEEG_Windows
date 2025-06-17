## ✅ `train_model.py` — *Model Training Pipeline*

| **Function / Component**     | **Description**                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `set_seed(seed=42)`          | Ensures reproducibility by setting seeds for PyTorch, NumPy, and random module.              |
| `EEGDatasetFromNumpy`        | Custom dataset class that loads EEG data from NumPy arrays and returns tensors.              |
| `train_loader`, `val_loader` | DataLoaders for batching and shuffling the train/validation datasets.                        |
| `MODEL_VERSION`              | A switch that allows you to select the model variant: `"imbalance"`, `"smote"`, or `"cost"`. |
| `compute_class_weight(...)`  | Computes class weights for imbalanced classification (only used in cost-sensitive training). |
| `criterion`                  | Loss function: CrossEntropyLoss. If cost-sensitive, it includes class weights.               |
| `optimizer`                  | Adam optimizer for updating model weights.                                                   |
| `scheduler`                  | Learning rate scheduler that decays the LR every few epochs.                                 |
| `Training Loop`              | Iterates over batches to perform forward, backward passes, and updates weights.              |
| `Validation Loop`            | Evaluates model on validation set and tracks best model based on val loss.                   |
| `torch.save(...)`            | Saves the model weights when it performs best on validation data.                            |

---

## ✅ `evaluate_model.py` — *Model Evaluation Pipeline*

| **Function / Component** | **Description**                                                    |
| ------------------------ | ------------------------------------------------------------------ |
| `EEGDatasetFromNumpy`    | Reuses the same dataset class to load test data from `.npy` files. |
| `test_loader`            | DataLoader for test set.                                           |
| `torch.load(...)`        | Loads the best saved model weights.                                |
| `model.eval()`           | Switches model to evaluation mode (turns off dropout etc.).        |
| `accuracy_score`         | Scikit-learn function to calculate overall accuracy.               |
| `confusion_matrix`       | Prints a 2x2 matrix showing true vs predicted labels.              |
| `classification_report`  | Provides precision, recall, F1-score per class.                    |
| `roc_auc_score`          | Computes the area under the ROC curve (binary classification).     |

---

## ✅ `preprocess_pipeline.py` — *Data Loader Module*

| **Class / Method**             | **Description**                                                  |
| ------------------------------ | ---------------------------------------------------------------- |
| `EEGDatasetFromNumpy`          | Custom PyTorch dataset class to work with `.npy` files directly. |
| `__init__(self, data, labels)` | Accepts NumPy arrays and converts them into PyTorch tensors.     |
| `__len__`                      | Returns the number of samples.                                   |
| `__getitem__`                  | Returns a `(data, label)` tuple for a given index.               |

---

## ✅ Model Files (all 3)

Each file defines a variant of the EEG Transformer model:

| **Model Class**            | **Description**                                                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EEGTransformer`           | Baseline transformer for imbalanced EEG classification.                                                                                           |
| `SMOTETransformer`         | Identical architecture, used with SMOTE-balanced datasets.                                                                                        |
| `CostSensitiveTransformer` | Also similar, used in conjunction with class-weighted loss.                                                                                       |
| **Shared Architecture**    | All models use:<br>• Linear layer to embed input<br>• Transformer Encoder<br>• Global average pooling<br>• Final classifier with dropout and ReLU |
