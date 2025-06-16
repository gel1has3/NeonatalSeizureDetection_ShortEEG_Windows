import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
import os

# Import models
from models.imbalancetransformer import EEGTransformer
from models.smotetransformer import SMOTETransformer
from models.costsenstitivetransformer import CostSensitiveTransformer

# Import dataset
from preprocess_pipeline import EEGDatasetFromNumpy

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
INPUT_SIZE = 256
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_data = np.load("data/train_data.npy")
train_labels = np.load("data/train_labels.npy")
val_data = np.load("data/val_data.npy")
val_labels = np.load("data/val_labels.npy")

# Dataset and Dataloader
train_dataset = EEGDatasetFromNumpy(train_data, train_labels)
val_dataset = EEGDatasetFromNumpy(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Choose model version
MODEL_VERSION = "imbalance"  # choose from: imbalance, smote, cost

if MODEL_VERSION == "imbalance":
    model = EEGTransformer(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
elif MODEL_VERSION == "smote":
    model = SMOTETransformer(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
elif MODEL_VERSION == "cost":
    model = CostSensitiveTransformer(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
else:
    raise ValueError("Invalid model version selected")

# Compute class weights for cost-sensitive learning
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

if MODEL_VERSION == "cost":
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model_{MODEL_VERSION}.pth")
        print(f"Model saved at epoch {epoch+1}")

