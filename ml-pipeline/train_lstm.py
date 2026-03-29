"""
LSTM Classifier Training Script

Trains the shallow LSTM model for quick issue type classification.
Uses cropped damage regions from the YOLO detector as input.

The LSTM processes CNN features as a sequence, enabling it to
capture spatial patterns in damage regions.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


ISSUE_TYPES = ["cracked_screen", "battery_swelling", "charging_port_damage"]


class DamageCropDataset(Dataset):
    """
    Dataset of cropped damage regions with issue type labels.

    Expected directory structure:
        crops/
        ├── cracked_screen/
        │   ├── img001.jpg
        │   └── ...
        ├── battery_swelling/
        │   └── ...
        └── charging_port_damage/
            └── ...
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(ISSUE_TYPES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), class_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class LSTMClassifierModel(nn.Module):
    """Shallow LSTM classifier (same architecture as backend)."""

    def __init__(self, num_classes=3, hidden_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        batch_size = features.size(0)
        features = features.view(batch_size, 64, -1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


def train(
    data_dir: str = "datasets/crops",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    output_path: str = "../backend/weights/lstm_classifier.pt",
):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = DamageCropDataset(data_dir, transform=transform)

    if len(dataset) == 0:
        print(f"No images found in {data_dir}. Creating synthetic dataset for demo...")
        _create_synthetic_dataset(data_dir)
        dataset = DamageCropDataset(data_dir, transform=transform)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = LSTMClassifierModel(num_classes=len(ISSUE_TYPES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss/len(train_loader):.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {output_path}")


def _create_synthetic_dataset(data_dir: str, num_per_class: int = 50):
    """Create synthetic placeholder images for demo/testing."""
    for class_name in ISSUE_TYPES:
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_per_class):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(class_dir, f"synth_{i:03d}.jpg"))
    print(f"Created synthetic dataset in {data_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM issue classifier")
    parser.add_argument("--data-dir", default="datasets/crops")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="../backend/weights/lstm_classifier.pt")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output)
