import json
import os
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dinov3_primary_classifier import DINOv3PrimaryClassifier


# Converts split records into primary-class training samples.
class PrimaryLabelDataset(Dataset):
    """Dataset wrapper that converts one-hot primary labels to class indices."""

    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Keep class order deterministic across train/val/test.
        self.primary_label_keys = sorted(list(data[0]["primary_label_one_hot"].keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        primary_label_dict = item["primary_label_one_hot"]
        # The JSON split stores labels as dict-based one-hot vectors, while
        # cross-entropy training expects an integer class index.
        primary_label_one_hot = torch.tensor(
            [primary_label_dict[k] for k in self.primary_label_keys], dtype=torch.float32
        )
        primary_label_index = torch.argmax(primary_label_one_hot).long()
        return image, primary_label_index


# Reads a saved JSON split file.
def load_split(path):
    """Load one of the precomputed JSON split files."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Builds train and validation loaders for primary classification.
def build_dataloaders(split_dir, train_batch_size, eval_batch_size, num_workers):
    """Create train/validation loaders that share one class ordering."""

    train_data = load_split(os.path.join(split_dir, "train.json"))
    val_data = load_split(os.path.join(split_dir, "val.json"))

    train_set = PrimaryLabelDataset(train_data)
    val_set = PrimaryLabelDataset(val_data)
    num_primary_labels = len(train_set.primary_label_keys)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return train_loader, val_loader, num_primary_labels


# Creates the primary model, loss, and optimizer.
def build_training_components(num_primary_labels, device, learning_rate):
    """Instantiate the model, loss, and optimizer used by the training loop."""

    model = DINOv3PrimaryClassifier(
        num_classes=num_primary_labels,

    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


# Trains the primary classifier and saves the best checkpoint.
def train_model(split_dir, epochs, checkpoint_dir, train_batch_size, eval_batch_size, num_workers, learning_rate):
    """Run training and save the best validation checkpoint."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_primary_labels = build_dataloaders(
        split_dir=split_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    model, criterion, optimizer = build_training_components(num_primary_labels, device, learning_rate)
    best_val_metric = -1.0
    best_path = os.path.join(checkpoint_dir, "best_model_primary.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        step = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, primary_label_indices in pbar:
            step += 1
            # Non-blocking transfers are effective because the DataLoader uses pin_memory.
            images = images.to(device, non_blocking=True)
            primary_label_indices = primary_label_indices.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, primary_label_indices)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / step
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Validation uses the same model weights without gradient updates.
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, primary_label_indices in val_loader:
                images = images.to(device, non_blocking=True)
                primary_label_indices = primary_label_indices.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, primary_label_indices)
                val_loss += loss.item()

                _, predicted_indices = torch.max(logits, 1)
                correct += (predicted_indices == primary_label_indices).sum().item()
                total += primary_label_indices.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} Validation Accuracy: {val_acc:.4f}  ValLoss: {avg_val_loss:.4f}")

        if val_acc > best_val_metric:
            best_val_metric = val_acc
            torch.save(model.state_dict(), best_path)
            print("best model saved")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_metric:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("Use evaluate_primary_classifier.py for final test-set evaluation.")


if __name__ == "__main__":
    freeze_support()
    train_model(
        split_dir="data_splits",
        epochs=25,
        checkpoint_dir="model",
        train_batch_size=64,
        eval_batch_size=32,
        num_workers=4,
        learning_rate=1e-4,
    )
