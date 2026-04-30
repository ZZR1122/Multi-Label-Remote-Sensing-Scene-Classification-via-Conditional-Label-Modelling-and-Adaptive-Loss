import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinov3_secondary_baselines import DINOv3BCEClassifier
from train_secondary_bce_baseline import SecondaryLabelDataset, load_split


# Loads the trained BCE baseline model for evaluation.
def build_model(num_secondary_labels, device, model_path):
    """Load the trained BCE baseline and switch it to eval mode."""

    model = DINOv3BCEClassifier(num_classes=num_secondary_labels).to(device)
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


# Computes sample-level Jaccard scores.
def sample_jaccard_batch(preds, targets):
    """Compute per-sample Jaccard scores for already binarized predictions."""

    inter = (preds & targets).astype(np.int32).sum(axis=1)
    union = (preds | targets).astype(np.int32).sum(axis=1)
    return np.where(union > 0, inter / union, 1.0)


# Handles ROC-AUC when a class has one target value.
def safe_roc_auc_score(y_true, y_score):
    """Return ROC-AUC when both classes are present, otherwise NaN."""

    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# Evaluates the BCE baseline model and writes CSV reports.
def test_model(
    model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    threshold,
):
    """Run BCE baseline evaluation and export summary, per-class metrics, and hard cases."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(report_dir, exist_ok=True)

    test_data = load_split(split_file)
    test_set = SecondaryLabelDataset(test_data)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = test_set.secondary_label_keys
    num_secondary_labels = len(class_names)
    model = build_model(num_secondary_labels, device, model_path)
    criterion = nn.BCEWithLogitsLoss()
    param_summary = summarize_parameters(model, "model")

    all_logits = []
    all_secondary_label_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for images, secondary_label_targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

            logits = model(images)
            total_loss += criterion(logits, secondary_label_targets).item()

            all_logits.append(logits.detach().cpu().float())
            all_secondary_label_targets.append(secondary_label_targets.detach().cpu().float())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_secondary_label_targets = torch.cat(all_secondary_label_targets, dim=0).numpy()
    avg_loss = total_loss / len(test_loader)

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds_bin = (probs >= threshold).astype(np.int32)
    targets_bin = (all_secondary_label_targets >= 0.5).astype(np.int32)

    sample_jaccards = sample_jaccard_batch(preds_bin, targets_bin)
    mean_sample_jaccard = float(sample_jaccards.mean())
    subset_accuracy = float(np.mean(np.all(preds_bin == targets_bin, axis=1)))
    label_cardinality = float(targets_bin.sum(axis=1).mean())
    pred_cardinality = float(preds_bin.sum(axis=1).mean())
    label_density = label_cardinality / num_secondary_labels

    per_class_jaccard = []
    for c in range(num_secondary_labels):
        p = preds_bin[:, c]
        t = targets_bin[:, c]
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        per_class_jaccard.append(inter / union if union > 0 else 1.0)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_bin, preds_bin, average=None, zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average="weighted", zero_division=0
    )
    sample_avg = precision_recall_fscore_support(
        targets_bin,
        preds_bin,
        average="samples",
        zero_division=0,
    )
    prec_samples, rec_samples, f1_samples, _ = sample_avg

    per_class_ap = []
    per_class_roc_auc = []
    for c in range(num_secondary_labels):
        y_true = all_secondary_label_targets[:, c]
        y_score = probs[:, c]
        per_class_ap.append(float(average_precision_score(y_true, y_score)))
        per_class_roc_auc.append(safe_roc_auc_score(y_true, y_score))

    map_mean = float(np.mean(per_class_ap))

    lrap = label_ranking_average_precision_score(all_secondary_label_targets, probs)
    coverage = coverage_error(all_secondary_label_targets, probs)
    ranking_loss = label_ranking_loss(all_secondary_label_targets, probs)
    hamming = hamming_loss(targets_bin, preds_bin)

    tp = ((preds_bin == 1) & (targets_bin == 1)).sum(axis=0)
    fp = ((preds_bin == 1) & (targets_bin == 0)).sum(axis=0)
    fn = ((preds_bin == 0) & (targets_bin == 1)).sum(axis=0)
    pos_rate = targets_bin.mean(axis=0)
    pred_pos_rate = preds_bin.mean(axis=0)

    rows = []
    for c in range(num_secondary_labels):
        rows.append(
            {
                "class_idx": c,
                "class_name": class_names[c],
                "support": int(support[c]),
                "pos_rate": float(pos_rate[c]),
                "pred_pos_rate": float(pred_pos_rate[c]),
                "tp": int(tp[c]),
                "fp": int(fp[c]),
                "fn": int(fn[c]),
                "jaccard": float(per_class_jaccard[c]),
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "avg_precision": float(per_class_ap[c]),
                "roc_auc": float(per_class_roc_auc[c]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(report_dir, "per_class_metrics.csv"), index=False)

    summary = {
        "model_path": model_path,
        "split_file": split_file,
        "num_samples": int(all_secondary_label_targets.shape[0]),
        "num_secondary_labels": int(num_secondary_labels),
        "threshold": float(threshold),
        "mean_sample_jaccard": mean_sample_jaccard,
        "subset_accuracy": subset_accuracy,
        "label_cardinality": label_cardinality,
        "pred_cardinality": pred_cardinality,
        "label_density": label_density,
        "prec_micro": float(prec_micro),
        "rec_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "prec_macro": float(prec_macro),
        "rec_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "prec_weighted": float(prec_weighted),
        "rec_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "prec_samples": float(prec_samples),
        "rec_samples": float(rec_samples),
        "f1_samples": float(f1_samples),
        "map": map_mean,
        "lrap": float(lrap),
        "coverage_error": float(coverage),
        "ranking_loss": float(ranking_loss),
        "hamming_loss": float(hamming),
        "avg_loss": float(avg_loss),
        "eval_loss_name": "BCEWithLogitsLoss",
        "avg_bce_loss": float(avg_loss),
        "experiment_name": "dinov3_bce",
        "backbone": "DINOv3",
        "loss": "BCEWithLogitsLoss",
        "correlation_modeling": 0,
    }
    summary.update(param_summary)
    pd.DataFrame([summary]).to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)

    print("Saved CSV reports to", report_dir)
    print("Summary:", summary)
    print("Evaluation complete.")
    return summary


# Runs evaluation with the default script settings.
def main(
    model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    threshold,
):
    """Entry point used by the module-level CLI."""

    return test_model(
        model_path=model_path,
        split_file=split_file,
        report_dir=report_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
    )


if __name__ == "__main__":
    main(
        model_path="ablation_models/dinov3_bce/best_model_multilabel_bce.pth",
        split_file="data_splits/test.json",
        report_dir="results/eval_report_multilabel_bce",
        batch_size=32,
        num_workers=4,
        threshold=0.5,
    )
