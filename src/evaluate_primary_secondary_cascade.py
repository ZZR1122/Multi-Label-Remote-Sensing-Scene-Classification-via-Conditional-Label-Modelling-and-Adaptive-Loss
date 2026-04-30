import json
import os

import numpy as np
import pandas as pd
import torch
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

from dinov3_conditional_multilabel import DINOv3ConditionalMultilabelClassifier
from dinov3_primary_classifier import DINOv3PrimaryClassifier
from train_conditional_multilabel_aslgb import PrimarySecondaryLabelDataset


# Reads a saved JSON split file.
def load_split(path):
    """Load a JSON split file for evaluation."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Loads the trained primary classifier for the cascade.
def build_primary_model(num_primary_labels, device, model_path):
    """Load the trained primary classifier and switch it to eval mode."""

    model = DINOv3PrimaryClassifier(num_classes=num_primary_labels).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


# Loads the trained conditional secondary model for the cascade.
def build_secondary_model(num_secondary_labels, num_primary_labels, device, model_path):
    """Load the trained conditional multilabel classifier and switch it to eval mode."""

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


# Evaluates the primary-to-secondary cascade and writes reports.
def test_model(
    primary_model_path,
    secondary_model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    threshold,
):
    """Run cascade evaluation and export both secondary-only and combined metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(report_dir, exist_ok=True)

    test_data = load_split(split_file)
    test_set = PrimarySecondaryLabelDataset(test_data)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_secondary_labels = len(test_set.secondary_label_keys)
    num_primary_labels = len(test_set.primary_label_keys)

    primary_model = build_primary_model(num_primary_labels, device, primary_model_path)
    secondary_model = build_secondary_model(num_secondary_labels, num_primary_labels, device, secondary_model_path)
    param_summary = summarize_parameters(primary_model, "primary_model")
    param_summary.update(summarize_parameters(secondary_model, "secondary_model"))
    param_summary["cascade_params_total"] = (
        param_summary["primary_model_params_total"] + param_summary["secondary_model_params_total"]
    )
    param_summary["cascade_params_trainable"] = (
        param_summary["primary_model_params_trainable"] + param_summary["secondary_model_params_trainable"]
    )

    all_secondary_logits = []
    all_secondary_label_targets = []
    all_primary_logits = []
    all_primary_preds = []
    all_primary_label_targets = []

    # The cascade first predicts the primary label, then feeds the predicted one-hot
    # vector into the secondary multilabel head.
    with torch.no_grad():
        for images, primary_label_targets, secondary_label_targets in tqdm(test_loader, desc="Evaluating cascade"):
            images = images.to(device, non_blocking=True)
            primary_label_targets = primary_label_targets.to(device, non_blocking=True)
            secondary_label_targets = secondary_label_targets.to(device, non_blocking=True)

            primary_logits = primary_model(images)
            primary_pred_indices = torch.argmax(primary_logits, dim=1)
            primary_true_indices = torch.argmax(primary_label_targets, dim=1)

            # The secondary model consumes the predicted primary class as a one-hot condition.
            primary_label_pred_one_hot = torch.zeros(
                (primary_pred_indices.shape[0], num_primary_labels), device=device, dtype=torch.float32
            )
            primary_label_pred_one_hot.scatter_(1, primary_pred_indices.unsqueeze(1), 1.0)

            secondary_logits = secondary_model(images, primary_label_pred_one_hot)

            all_secondary_logits.append(secondary_logits.detach().cpu().float())
            all_secondary_label_targets.append(secondary_label_targets.detach().cpu().float())
            all_primary_logits.append(primary_logits.detach().cpu().float())
            all_primary_preds.append(primary_pred_indices.detach().cpu())
            all_primary_label_targets.append(primary_true_indices.detach().cpu())

    all_secondary_logits = torch.cat(all_secondary_logits, dim=0).numpy()
    all_secondary_label_targets = torch.cat(all_secondary_label_targets, dim=0).numpy()
    all_primary_logits = torch.cat(all_primary_logits, dim=0).numpy()
    all_primary_preds = torch.cat(all_primary_preds, dim=0).numpy()
    all_primary_label_targets = torch.cat(all_primary_label_targets, dim=0).numpy()

    primary_probs = torch.softmax(torch.from_numpy(all_primary_logits), dim=1).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_secondary_logits))
    preds_bin = (probs >= threshold).astype(np.int32)
    targets_bin = (all_secondary_label_targets >= 0.5).astype(np.int32)

    sample_jaccards = sample_jaccard_batch(preds_bin, targets_bin)
    mean_sample_jaccard = float(sample_jaccards.mean())
    subset_accuracy = float(np.mean(np.all(preds_bin == targets_bin, axis=1)))
    label_cardinality = float(targets_bin.sum(axis=1).mean())
    pred_cardinality = float(preds_bin.sum(axis=1).mean())
    label_density = label_cardinality / num_secondary_labels

    primary_correct_mask = all_primary_preds == all_primary_label_targets
    secondary_subset_mask = np.all(preds_bin == targets_bin, axis=1)
    secondary_any_true_mask = targets_bin.sum(axis=1) > 0
    secondary_any_match_mask = (preds_bin & targets_bin).sum(axis=1) > 0
    secondary_empty_mask = ~secondary_any_true_mask
    secondary_empty_pred_mask = preds_bin.sum(axis=1) == 0

    joint_strict_accuracy = float(np.mean(primary_correct_mask & secondary_subset_mask))
    joint_primary_and_any_match = float(np.mean(primary_correct_mask & secondary_any_match_mask))
    secondary_any_match_rate = float(np.mean(secondary_any_match_mask))
    secondary_any_match_rate_pos = float(np.mean(secondary_any_match_mask[secondary_any_true_mask]))
    secondary_empty_match_rate = float(np.mean(secondary_empty_pred_mask[secondary_empty_mask]))

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
        ap = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        per_class_ap.append(ap)
        per_class_roc_auc.append(roc_auc)

    map_mean = float(np.mean(per_class_ap))

    lrap = label_ranking_average_precision_score(all_secondary_label_targets, probs)
    coverage = coverage_error(all_secondary_label_targets, probs)
    ranking_loss = label_ranking_loss(all_secondary_label_targets, probs)
    hamming = hamming_loss(targets_bin, preds_bin)

    tp = ((preds_bin == 1) & (targets_bin == 1)).sum(axis=0)
    fp = ((preds_bin == 1) & (targets_bin == 0)).sum(axis=0)
    fn = ((preds_bin == 0) & (targets_bin == 1)).sum(axis=0)
    num_samples = targets_bin.shape[0]
    tn = num_samples - tp - fp - fn
    acc_micro = float(np.mean(preds_bin == targets_bin))
    acc_samples = float(np.mean((preds_bin == targets_bin).mean(axis=1)))
    acc_macro = float(np.mean((tp + tn) / num_samples))
    pos_rate = targets_bin.mean(axis=0)
    pred_pos_rate = preds_bin.mean(axis=0)

    rows = []
    for c in range(num_secondary_labels):
        rows.append(
            {
                "class_idx": c,
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

    pd.DataFrame(rows).to_csv(os.path.join(report_dir, "per_class_metrics.csv"), index=False)

    summary = {
        "primary_model_path": primary_model_path,
        "secondary_model_path": secondary_model_path,
        "split_file": split_file,
        "num_samples": int(all_secondary_label_targets.shape[0]),
        "num_secondary_labels": int(num_secondary_labels),
        "threshold": float(threshold),
        "mean_sample_jaccard": mean_sample_jaccard,
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
    }
    summary.update(
        {
            "joint_strict_accuracy": joint_strict_accuracy,
            "joint_primary_and_any_match": joint_primary_and_any_match,
            "secondary_any_match_rate": secondary_any_match_rate,
            "secondary_any_match_rate_pos": secondary_any_match_rate_pos,
            "secondary_empty_match_rate": secondary_empty_match_rate,
        }
    )
    summary.update(param_summary)

    primary_true_bin = np.zeros((all_primary_label_targets.shape[0], num_primary_labels), dtype=np.int32)
    primary_pred_bin = np.zeros((all_primary_preds.shape[0], num_primary_labels), dtype=np.int32)
    primary_true_bin[np.arange(all_primary_label_targets.shape[0]), all_primary_label_targets] = 1
    primary_pred_bin[np.arange(all_primary_preds.shape[0]), all_primary_preds] = 1

    # The combined report concatenates primary and secondary labels into one
    # joint label space so the whole cascade can be evaluated end to end.
    combined_targets = np.concatenate([primary_true_bin, targets_bin], axis=1)
    combined_preds = np.concatenate([primary_pred_bin, preds_bin], axis=1)
    combined_scores = np.concatenate([primary_probs, probs], axis=1)
    combined_num_classes = combined_targets.shape[1]

    combined_sample_jaccards = sample_jaccard_batch(combined_preds, combined_targets)
    combined_mean_sample_jaccard = float(combined_sample_jaccards.mean())
    combined_subset_accuracy = float(np.mean(np.all(combined_preds == combined_targets, axis=1)))
    combined_label_cardinality = float(combined_targets.sum(axis=1).mean())
    combined_pred_cardinality = float(combined_preds.sum(axis=1).mean())
    combined_label_density = combined_label_cardinality / combined_num_classes

    comb_precision, comb_recall, comb_f1, comb_support = precision_recall_fscore_support(
        combined_targets, combined_preds, average=None, zero_division=0
    )
    comb_prec_micro, comb_rec_micro, comb_f1_micro, _ = precision_recall_fscore_support(
        combined_targets, combined_preds, average="micro", zero_division=0
    )
    comb_prec_macro, comb_rec_macro, comb_f1_macro, _ = precision_recall_fscore_support(
        combined_targets, combined_preds, average="macro", zero_division=0
    )
    comb_prec_weighted, comb_rec_weighted, comb_f1_weighted, _ = precision_recall_fscore_support(
        combined_targets, combined_preds, average="weighted", zero_division=0
    )
    comb_sample_avg = precision_recall_fscore_support(
        combined_targets,
        combined_preds,
        average="samples",
        zero_division=0,
    )
    comb_prec_samples, comb_rec_samples, comb_f1_samples, _ = comb_sample_avg

    comb_per_class_ap = []
    comb_per_class_roc_auc = []
    for c in range(combined_num_classes):
        y_true = combined_targets[:, c]
        y_score = combined_scores[:, c]
        ap = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        comb_per_class_ap.append(ap)
        comb_per_class_roc_auc.append(roc_auc)

    combined_map = float(np.mean(comb_per_class_ap))
    combined_lrap = label_ranking_average_precision_score(combined_targets, combined_scores)
    combined_coverage = coverage_error(combined_targets, combined_scores)
    combined_ranking_loss = label_ranking_loss(combined_targets, combined_scores)
    combined_hamming = hamming_loss(combined_targets, combined_preds)

    comb_tp = ((combined_preds == 1) & (combined_targets == 1)).sum(axis=0)
    comb_fp = ((combined_preds == 1) & (combined_targets == 0)).sum(axis=0)
    comb_fn = ((combined_preds == 0) & (combined_targets == 1)).sum(axis=0)
    comb_num_samples = combined_targets.shape[0]
    comb_tn = comb_num_samples - comb_tp - comb_fp - comb_fn
    comb_acc_micro = float(np.mean(combined_preds == combined_targets))
    comb_acc_samples = float(np.mean((combined_preds == combined_targets).mean(axis=1)))
    comb_acc_macro = float(np.mean((comb_tp + comb_tn) / comb_num_samples))
    comb_pos_rate = combined_targets.mean(axis=0)
    comb_pred_pos_rate = combined_preds.mean(axis=0)

    combined_rows = []
    for c in range(combined_num_classes):
        if c < num_primary_labels:
            label_type = "primary"
            label_name = test_set.primary_label_keys[c]
            local_idx = c
        else:
            label_type = "secondary"
            local_idx = c - num_primary_labels
            label_name = test_set.secondary_label_keys[local_idx]
        combined_rows.append(
            {
                "global_idx": c,
                "label_type": label_type,
                "local_idx": local_idx,
                "label_name": label_name,
                "support": int(comb_support[c]),
                "pos_rate": float(comb_pos_rate[c]),
                "pred_pos_rate": float(comb_pred_pos_rate[c]),
                "tp": int(comb_tp[c]),
                "fp": int(comb_fp[c]),
                "fn": int(comb_fn[c]),
                "precision": float(comb_precision[c]),
                "recall": float(comb_recall[c]),
                "f1": float(comb_f1[c]),
                "avg_precision": float(comb_per_class_ap[c]),
                "roc_auc": float(comb_per_class_roc_auc[c]),
            }
        )

    pd.DataFrame(combined_rows).to_csv(
        os.path.join(report_dir, "combined_per_class_metrics.csv"), index=False
    )

    summary.update(
        {
            "combined_num_classes": int(combined_num_classes),
            "combined_mean_sample_jaccard": combined_mean_sample_jaccard,
            "combined_subset_accuracy": combined_subset_accuracy,
            "combined_label_cardinality": combined_label_cardinality,
            "combined_pred_cardinality": combined_pred_cardinality,
            "combined_label_density": combined_label_density,
            "combined_prec_micro": float(comb_prec_micro),
            "combined_rec_micro": float(comb_rec_micro),
            "combined_f1_micro": float(comb_f1_micro),
            "combined_prec_macro": float(comb_prec_macro),
            "combined_rec_macro": float(comb_rec_macro),
            "combined_f1_macro": float(comb_f1_macro),
            "combined_prec_weighted": float(comb_prec_weighted),
            "combined_rec_weighted": float(comb_rec_weighted),
            "combined_f1_weighted": float(comb_f1_weighted),
            "combined_prec_samples": float(comb_prec_samples),
            "combined_rec_samples": float(comb_rec_samples),
            "combined_f1_samples": float(comb_f1_samples),
            "combined_map": combined_map,
            "combined_lrap": float(combined_lrap),
            "combined_coverage_error": float(combined_coverage),
            "combined_ranking_loss": float(combined_ranking_loss),
            "combined_hamming_loss": float(combined_hamming),
        }
    )

    compare_columns = [
        "scope",
        "primary_model_path",
        "secondary_model_path",
        "split_file",
        "primary_model_params_total",
        "primary_model_params_trainable",
        "secondary_model_params_total",
        "secondary_model_params_trainable",
        "cascade_params_total",
        "cascade_params_trainable",
        "num_samples",
        "num_classes",
        "threshold",
        "acc_micro",
        "acc_macro",
        "acc_samples",
        "mean_sample_jaccard",
        "label_cardinality",
        "pred_cardinality",
        "label_density",
        "prec_micro",
        "rec_micro",
        "f1_micro",
        "prec_macro",
        "rec_macro",
        "f1_macro",
        "prec_weighted",
        "rec_weighted",
        "f1_weighted",
        "prec_samples",
        "rec_samples",
        "f1_samples",
        "map",
        "lrap",
        "coverage_error",
        "ranking_loss",
        "hamming_loss",
    ]

    secondary_row = {
        "scope": "secondary",
        "primary_model_path": primary_model_path,
        "secondary_model_path": secondary_model_path,
        "split_file": split_file,
        "num_samples": int(all_secondary_label_targets.shape[0]),
        "num_classes": int(num_secondary_labels),
        "threshold": float(threshold),
        "acc_micro": acc_micro,
        "acc_macro": acc_macro,
        "acc_samples": acc_samples,
        "mean_sample_jaccard": mean_sample_jaccard,
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
    }
    secondary_row.update(param_summary)

    combined_row = {
        "scope": "combined",
        "primary_model_path": primary_model_path,
        "secondary_model_path": secondary_model_path,
        "split_file": split_file,
        "num_samples": int(all_secondary_label_targets.shape[0]),
        "num_classes": int(combined_num_classes),
        "threshold": float(threshold),
        "acc_micro": comb_acc_micro,
        "acc_macro": comb_acc_macro,
        "acc_samples": comb_acc_samples,
        "mean_sample_jaccard": combined_mean_sample_jaccard,
        "label_cardinality": combined_label_cardinality,
        "pred_cardinality": combined_pred_cardinality,
        "label_density": combined_label_density,
        "prec_micro": float(comb_prec_micro),
        "rec_micro": float(comb_rec_micro),
        "f1_micro": float(comb_f1_micro),
        "prec_macro": float(comb_prec_macro),
        "rec_macro": float(comb_rec_macro),
        "f1_macro": float(comb_f1_macro),
        "prec_weighted": float(comb_prec_weighted),
        "rec_weighted": float(comb_rec_weighted),
        "f1_weighted": float(comb_f1_weighted),
        "prec_samples": float(comb_prec_samples),
        "rec_samples": float(comb_rec_samples),
        "f1_samples": float(comb_f1_samples),
        "map": combined_map,
        "lrap": float(combined_lrap),
        "coverage_error": float(combined_coverage),
        "ranking_loss": float(combined_ranking_loss),
        "hamming_loss": float(combined_hamming),
    }
    combined_row.update(param_summary)

    compare_df = pd.DataFrame([secondary_row, combined_row])
    compare_df = compare_df.reindex(columns=compare_columns)
    compare_df.to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)

    print("Saved reports to", report_dir)
    print("Summary:", summary)
    return summary


# Runs cascade evaluation with the default script settings.
def main(
    primary_model_path,
    secondary_model_path,
    split_file,
    report_dir,
    batch_size,
    num_workers,
    threshold,
):
    """Entry point used by the module-level CLI."""

    return test_model(
        primary_model_path=primary_model_path,
        secondary_model_path=secondary_model_path,
        split_file=split_file,
        report_dir=report_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
    )


if __name__ == "__main__":
    main(
        primary_model_path="model/best_model_primary.pth",
        secondary_model_path="model/best_model_conditional_multilabel_aslgb.pth",
        split_file="data_splits/test.json",
        report_dir="results/eval_report_cascade",
        batch_size=32,
        num_workers=4,
        threshold=0.5,
    )
