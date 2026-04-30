import json
import os
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dinov3_secondary_baselines import DINOv3BCEClassifier


# Converts split records into secondary-label samples.
class SecondaryLabelDataset(Dataset):
    """Return an image together with its secondary multilabel targets."""

    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.secondary_label_keys = sorted(data[0]["secondary_labels"].keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        secondary_label_targets = torch.tensor(
            [item["secondary_labels"][k] for k in self.secondary_label_keys],
            dtype=torch.float32,
        )
        return image, secondary_label_targets


# Reads a saved JSON split file.
def load_split(path):
    """Load one of the JSON split files produced by `dataset_splitter.py`."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Builds train and validation loaders for secondary baselines.
def build_dataloaders(split_dir, train_batch_size, eval_batch_size, num_workers):
    """Create train/validation loaders with consistent label ordering."""

    train_data = load_split(os.path.join(split_dir, "train.json"))
    val_data = load_split(os.path.join(split_dir, "val.json"))

    train_set = SecondaryLabelDataset(train_data)
    val_set = SecondaryLabelDataset(val_data)

    num_secondary_labels = len(train_set.secondary_label_keys)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, num_secondary_labels


# Creates the BCE baseline model, loss, and optimizer.
def build_training_components(num_secondary_labels, device, learning_rate, weight_decay):
    """Instantiate the BCE baseline, loss, and optimizer used by the training loop."""

    model = DINOv3BCEClassifier(num_classes=num_secondary_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, criterion, optimizer


# Computes validation metrics for multilabel predictions.
def multilabel_metrics(preds, targets):
    """Compute the core multilabel metrics used during validation and testing."""

    preds = preds.int()
    targets = targets.int()

    tp = (preds & targets).sum(dim=0)
    fp = (preds & (1 - targets)).sum(dim=0)
    fn = ((1 - preds) & targets).sum(dim=0)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_f1 = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn + 1e-8)

    f1_per_class = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    macro_f1 = f1_per_class.mean()

    hamming_loss = (preds != targets).float().mean()
    subset_acc = (preds == targets).all(dim=1).float().mean()

    return (
        micro_f1.item(),
        macro_f1.item(),
        hamming_loss.item(),
        subset_acc.item(),
    )


# Trains the BCE secondary-label baseline.
def train_model(
    split_dir,
    epochs,
    checkpoint_path,
    train_batch_size,
    eval_batch_size,
    num_workers,
    learning_rate,
    weight_decay,
):
    """Run BCE baseline training and save the best validation checkpoint."""

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_secondary_labels = build_dataloaders(
        split_dir=split_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    model, criterion, optimizer = build_training_components(
        num_secondary_labels=num_secondary_labels,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_metric = -1.0
    checkpoint_label = checkpoint_path

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, secondary_label_targets in pbar:
            images = images.to(device, non_blocking=True)
            secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, secondary_label_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        model.eval()
        val_loss = 0.0

        all_preds = []
        all_secondary_label_targets = []

        with torch.no_grad():
            for images, secondary_label_targets in val_loader:
                images = images.to(device, non_blocking=True)
                secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, secondary_label_targets)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = probs > 0.5

                all_preds.append(preds.cpu())
                all_secondary_label_targets.append(secondary_label_targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_secondary_label_targets = torch.cat(all_secondary_label_targets, dim=0)

        (
            val_micro_f1,
            val_macro_f1,
            val_hamming,
            val_subset_acc,
        ) = multilabel_metrics(all_preds, all_secondary_label_targets)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Micro-F1: {val_micro_f1:.4f}  "
            f"Macro-F1: {val_macro_f1:.4f}  "
            f"Hamming: {val_hamming:.4f}  "
            f"SubsetAcc: {val_subset_acc:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if val_micro_f1 > best_val_metric:
            best_val_metric = val_micro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print("Save best model")

    print("\nTraining complete.")
    print(f"Best validation Micro-F1: {best_val_metric:.4f}")
    print(f"Best checkpoint: {checkpoint_label}")
    print("Use evaluate_secondary_bce_baseline.py for final test-set evaluation.")


if __name__ == "__main__":
    freeze_support()
    train_model(
        split_dir="data_splits",
        epochs=25,
        checkpoint_path="ablation_models/dinov3_bce/best_model_multilabel_bce.pth",
        train_batch_size=64,
        eval_batch_size=32,
        num_workers=4,
        learning_rate=1e-4,
        weight_decay=0.0,
    )
