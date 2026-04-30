import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinov3_primary_classifier import DINOv3PrimaryClassifier
from train_primary_classifier import PrimaryLabelDataset


# Reads a saved JSON split file.
def load_split(path):
    """Load a JSON split file for evaluation."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Loads the trained primary classifier for evaluation.
def build_model(num_primary_labels, device, model_path):
    """Load the trained primary classifier checkpoint and switch it to eval mode."""

    model = DINOv3PrimaryClassifier(
        num_classes=num_primary_labels,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


# Counts total and trainable parameters for reports.
def summarize_parameters(module, prefix):
    """Return total and trainable parameter counts for reporting."""

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        f"{prefix}_params_total": int(total),
        f"{prefix}_params_trainable": int(trainable),
    }


# Evaluates primary classification and writes CSV reports.
def test_model(
    model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    top_k_list,
):
    """Run primary-label evaluation and write summary and per-class reports."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(report_dir, exist_ok=True)

    test_data = load_split(split_file)
    test_set = PrimaryLabelDataset(test_data)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    num_primary_labels = len(test_set.primary_label_keys)
    primary_label_names = test_set.primary_label_keys

    model = build_model(num_primary_labels, device, model_path)
    param_summary = summarize_parameters(model, "model")

    all_logits = []
    all_primary_label_indices = []

    # Evaluation stores the full logit matrix so all downstream metrics can be
    # computed from the same pass through the test set.
    with torch.no_grad():
        for images, primary_label_indices in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            primary_label_indices = primary_label_indices.to(device, non_blocking=True)

            logits = model(images)
            all_logits.append(logits.detach().cpu().float())
            all_primary_label_indices.append(primary_label_indices.detach().cpu().long())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_primary_label_indices = torch.cat(all_primary_label_indices, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    all_preds = np.argmax(all_logits, axis=1)

    accuracy = float(np.mean(all_preds == all_primary_label_indices))
    topk_scores = {
        f"top_{k}": float(
            top_k_accuracy_score(all_primary_label_indices, probs, k=k, labels=np.arange(probs.shape[1]))
        )
        for k in top_k_list
    }

    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        all_primary_label_indices, all_preds, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_primary_label_indices, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_primary_label_indices, all_preds, average="weighted", zero_division=0
    )

    summary = {
        "model_path": model_path,
        "split_file": split_file,
        "num_samples": int(all_primary_label_indices.shape[0]),
        "num_primary_labels": int(num_primary_labels),
        "accuracy": accuracy,
        "prec_micro": float(prec_micro),
        "rec_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "prec_macro": float(prec_macro),
        "rec_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "prec_weighted": float(prec_weighted),
        "rec_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
    }
    summary.update({k: v for k, v in topk_scores.items()})
    summary.update(param_summary)

    pd.DataFrame([summary]).to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)

    print("Saved reports to", report_dir)
    print("Summary:", summary)
    return summary


# Runs evaluation with the default script settings.
def main(
    model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    top_k_list,
):
    """Entry point used by the module-level CLI."""

    return test_model(
        model_path=model_path,
        split_file=split_file,
        report_dir=report_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k_list=top_k_list,
    )


if __name__ == "__main__":
    model_paths = ["model/best_model_primary.pth"]
    for model_path in model_paths:
        print(f"Evaluating model: {model_path}")
        main(
            model_path=model_path,
            split_file="data_splits/test.json",
            report_dir="results/eval_report_primary",
            batch_size=32,
            num_workers=4,
            top_k_list=(3, 5),
        )
