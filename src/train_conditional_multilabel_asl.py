import json
import os
from multiprocessing import freeze_support

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import multilabel_losses
from dinov3_conditional_multilabel import DINOv3ConditionalMultilabelClassifier


# Converts split records into conditional multilabel samples.
class PrimarySecondaryLabelDataset(Dataset):
    """Return an image together with its primary one-hot vector and secondary targets."""

    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.secondary_label_keys = sorted(data[0]["secondary_labels"].keys())
        self.primary_label_keys = sorted(data[0]["primary_label_one_hot"].keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        # Secondary targets stay multi-hot because the task is multilabel.
        secondary_label_targets = torch.tensor(
            [item["secondary_labels"][k] for k in self.secondary_label_keys],
            dtype=torch.float32
        )

        # Primary labels are passed to the model as a conditioning signal.
        primary_label_targets = torch.tensor(
            [item["primary_label_one_hot"][k] for k in self.primary_label_keys],
            dtype=torch.float32
        )

        return image, primary_label_targets, secondary_label_targets


# Reads a saved JSON split file.
def load_split(path):
    """Load one of the JSON split files produced by `dataset_splitter.py`."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Builds train and validation loaders for conditional training.
def build_dataloaders(split_dir, train_batch_size, eval_batch_size, num_workers):
    """Create train/validation loaders with consistent label ordering."""

    train_data = load_split(os.path.join(split_dir, "train.json"))
    val_data = load_split(os.path.join(split_dir, "val.json"))

    train_set = PrimarySecondaryLabelDataset(train_data)
    val_set = PrimarySecondaryLabelDataset(val_data)

    num_secondary_labels = len(train_set.secondary_label_keys)
    num_primary_labels = len(train_set.primary_label_keys)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, num_secondary_labels, num_primary_labels


# Creates the conditional model, loss, and optimizer.
def build_training_components(num_secondary_labels, num_primary_labels, device, learning_rate):
    """Instantiate the conditional classifier, asymmetric loss, and optimizer."""

    model = DINOv3ConditionalMultilabelClassifier(
        num_classes=num_secondary_labels,
        cond_dim=num_primary_labels,
        label_embed_dim=384,
        num_heads=8,
        num_layers=3,
        image_tokens=None,
        cond_tokens=8,
        dropout=0.1,
    ).to(device)
    criterion = multilabel_losses.AsymmetricLossOptimized()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


# Computes validation metrics for multilabel predictions.
def multilabel_metrics(preds, targets):
    """Compute the core multilabel metrics used during validation and testing."""

    preds = preds.int()
    targets = targets.int()

    tp = (preds & targets).sum(dim=0)
    fp = (preds & (1 - targets)).sum(dim=0)
    fn = ((1 - preds) & targets).sum(dim=0)

    # Micro-F1 pools all class decisions before computing precision/recall balance.
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_f1 = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn + 1e-8)

    # Macro-F1 treats every class equally regardless of frequency.
    f1_per_class = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    macro_f1 = f1_per_class.mean()

    # Hamming loss measures the fraction of incorrect binary label decisions.
    hamming_loss = (preds != targets).float().mean()

    # Subset accuracy requires the entire label set of a sample to match exactly.
    subset_acc = (preds == targets).all(dim=1).float().mean()

    return (
        micro_f1.item(),
        macro_f1.item(),
        hamming_loss.item(),
        subset_acc.item(),
    )


# Trains the conditional secondary-label model.
def train_model(split_dir, epochs, checkpoint_dir, train_batch_size, eval_batch_size, num_workers, learning_rate):
    """Run training and keep the checkpoint with the best validation Micro-F1."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_secondary_labels, num_primary_labels = build_dataloaders(
        split_dir=split_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    model, criterion, optimizer = build_training_components(
        num_secondary_labels,
        num_primary_labels,
        device,
        learning_rate,
    )

    best_val_metric = -1.0
    best_path = os.path.join(checkpoint_dir, "best_model_conditional_multilabel_asl.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, primary_label_targets, secondary_label_targets in pbar:
            # The image and primary-label one-hot vector are both part of the model input.
            images = images.to(device, non_blocking=True)
            primary_label_targets = primary_label_targets.to(device, non_blocking=True)
            secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images, primary_label_targets)

            loss = criterion(logits, secondary_label_targets)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # Validation collects all predictions first, then computes dataset-level metrics.
        model.eval()
        val_loss = 0.0

        all_preds = []
        all_secondary_label_targets = []

        with torch.no_grad():
            for images, primary_label_targets, secondary_label_targets in val_loader:
                images = images.to(device, non_blocking=True)
                primary_label_targets = primary_label_targets.to(device, non_blocking=True)
                secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

                logits = model(images, primary_label_targets)
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

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Micro-F1: {val_micro_f1:.4f}  "
            f"Macro-F1: {val_macro_f1:.4f}  "
            f"Hamming: {val_hamming:.4f}  "
            f"SubsetAcc: {val_subset_acc:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # The validation Micro-F1 is the checkpoint selection criterion.
        if val_micro_f1 > best_val_metric:
            best_val_metric = val_micro_f1
            torch.save(model.state_dict(), best_path)
            print("Save best model")

    print("\nTraining complete.")
    print(f"Best validation Micro-F1: {best_val_metric:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("Use evaluate_conditional_multilabel_asl.py for final test-set evaluation.")


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
