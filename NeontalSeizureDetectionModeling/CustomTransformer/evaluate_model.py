import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from torch.utils.data import DataLoader

from models.imbalancetransformer import EEGTransformer
from preprocess_pipeline import EEGDatasetFromNumpy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_data = np.load("data/test_data.npy")
test_labels = np.load("data/test_labels.npy")

test_dataset = EEGDatasetFromNumpy(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = EEGTransformer(input_size=256, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("best_model_imbalance.pth"))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(DEVICE)
        outputs = model(batch_data)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_labels.numpy())

accuracy = accuracy_score(all_targets, all_preds)
conf_mat = confusion_matrix(all_targets, all_preds)
report = classification_report(all_targets, all_preds)
roc_auc = roc_auc_score(all_targets, all_preds)

print("Test Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", report)
print("ROC AUC:", roc_auc)
