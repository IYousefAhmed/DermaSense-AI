import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# =====================================================
# 1️⃣ FIXED CLASS ORDER (CRITICAL)
# =====================================================
CLASS_ORDER = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
label_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}

# =====================================================
# 2️⃣ LOAD METADATA
# =====================================================
metadata = pd.read_csv("HAM10000_metadata.csv")
metadata['label'] = metadata['dx'].map(label_map)

# Remove any rows with unknown labels (safety)
metadata = metadata.dropna(subset=['label'])

# =====================================================
# 3️⃣ STRATIFIED SPLIT
# =====================================================
train_df, temp_df = train_test_split(
    metadata,
    test_size=0.3,
    stratify=metadata['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label'],
    random_state=42
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# =====================================================
# 4️⃣ TRANSFORMS
# =====================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# 5️⃣ CUSTOM DATASET
# =====================================================
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['image_id'] + ".jpg"

        img_path = None
        for d in self.img_dirs:
            candidate = os.path.join(d, img_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"{img_name} not found.")

        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# 6️⃣ DATA LOADERS
# =====================================================
img_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

train_dataset = HAM10000Dataset(train_df, img_dirs, train_transform)
val_dataset = HAM10000Dataset(val_df, img_dirs, val_transform)
test_dataset = HAM10000Dataset(test_df, img_dirs, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Data loaders ready.")

# =====================================================
# 7️⃣ MODEL SETUP
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_ORDER))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# =====================================================
# 8️⃣ TRAINING LOOP (BEST MODEL SAVE)
# =====================================================
num_epochs = 5
best_val_acc = 0

for epoch in range(num_epochs):

    # ---- Training ----
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "../models/skin_cancer_resnet50.pth")
        print("Best model saved.")

print("Training complete.")

# =====================================================
# 9️⃣ TEST EVALUATION
# =====================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_ORDER))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("Final model saved in models/ folder.")