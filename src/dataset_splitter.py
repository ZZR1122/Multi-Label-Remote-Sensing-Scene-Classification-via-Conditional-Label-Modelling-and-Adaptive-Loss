import json
import os
import random
from collections import Counter

from dataset_loader import DataLoad


# Counts primary labels in a split.
def _count_primary(data):
    """Count primary-label frequencies."""

    counts = Counter()
    for item in data:
        counts[item["primary_label"]] += 1
    return counts


# Counts positive secondary labels in a split.
def _count_secondary(data):
    """Count positive occurrences of each secondary label."""

    counts = Counter()
    for item in data:
        for label_name, value in item["secondary_labels"].items():
            if value == 1:
                counts[label_name] += 1
    return counts


# Builds distribution-bias rows for split diagnostics.
def _build_bias_table(splits_counts, splits_sizes):
    """Build percentage/bias rows for each label across splits.

    Args:
        splits_counts: dict[split_name, Counter[label -> count]].
        splits_sizes: dict[split_name, int].
            If a `total` entry is present, it is treated as the reference dataset
            rather than an additional split to be summed again.

    Returns:
        list[dict]: Each row includes split percentages and bias against global percentage.
    """

    split_names = [name for name in ["train", "val", "test"] if name in splits_counts]
    reference_name = "total" if "total" in splits_counts else None

    labels = set()
    for split_name in split_names:
        labels.update(splits_counts[split_name].keys())
    if reference_name is not None:
        labels.update(splits_counts[reference_name].keys())
    labels = sorted(labels)

    if reference_name is not None:
        total_size = splits_sizes[reference_name]
        total_counts = {
            label: splits_counts[reference_name][label]
            for label in labels
        }
    else:
        total_size = sum(splits_sizes[split_name] for split_name in split_names)
        total_counts = {
            label: sum(splits_counts[split_name][label] for split_name in split_names)
            for label in labels
        }

    rows = []
    for label in labels:
        row = {"label": label}
        total_pct = total_counts[label] / total_size * 100

        for split_name in split_names:
            count = splits_counts[split_name][label]
            split_size = splits_sizes[split_name]
            split_pct = count / split_size * 100
            bias = split_pct - total_pct

            row[f"{split_name}_pct"] = split_pct
            row[f"{split_name}_bias"] = bias
            row[f"{split_name}_count"] = count

        row["total_pct"] = total_pct
        row["total_count"] = total_counts[label]
        rows.append(row)

    return rows


# Prints split-bias diagnostics in table form.
def _print_bias_table(rows, title):
    """Print an ASCII table and return summary statistics."""

    headers = ["Label", "Train (%)", "Bias", "Val (%)", "Bias", "Test (%)", "Bias"]

    table = []
    for row in rows:
        table.append(
            [
                row["label"],
                f"{row['train_pct']:.1f}%",
                f"{row['train_bias']:+.1f}",
                f"{row['val_pct']:.1f}%",
                f"{row['val_bias']:+.1f}",
                f"{row['test_pct']:.1f}%",
                f"{row['test_bias']:+.1f}",
            ]
        )

    col_widths = []
    for col_idx in range(len(headers)):
        max_len = max(len(str(row[col_idx])) for row in table + [headers])
        col_widths.append(max_len)

    print()
    print(title)

    header_row = "   ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    print(header_row)

    for row in table:
        line = "   ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))
        print(line)

    print()

    all_biases = []
    for row in rows:
        all_biases.extend([row["train_bias"], row["val_bias"], row["test_bias"]])

    max_abs_bias = max(abs(bias) for bias in all_biases)
    avg_abs_bias = sum(abs(bias) for bias in all_biases) / len(all_biases)

    print("Bias summary:")
    print(f"Max absolute bias: {max_abs_bias:.2f}%")
    print(f"Mean absolute bias: {avg_abs_bias:.2f}%")

    return {
        "max_absolute_bias": max_abs_bias,
        "avg_absolute_bias": avg_abs_bias,
        "rows": rows,
    }


# Compares primary and secondary label balance across splits.
def compare_and_print_bias_tables(splits):
    """Compare split distributions against the full dataset.

    Args:
        splits: dict with keys `train`, `val`, `test`, `total` mapping to sample lists.
    """

    sizes = {name: len(items) for name, items in splits.items()}

    # Primary and secondary labels are summarized independently so their bias can
    # be inspected without mixing single-label and multi-label statistics.
    splits_primary_counts = {name: _count_primary(items) for name, items in splits.items()}
    splits_secondary_counts = {name: _count_secondary(items) for name, items in splits.items()}

    primary_rows = _build_bias_table(splits_primary_counts, sizes)
    secondary_rows = _build_bias_table(splits_secondary_counts, sizes)

    primary_stats = _print_bias_table(
        primary_rows,
        "Primary Label Distribution Bias (Percentage and Deviation from Overall)",
    )
    secondary_stats = _print_bias_table(
        secondary_rows,
        "Secondary Label Distribution Bias (Percentage and Deviation from Overall)",
    )

    return primary_rows, secondary_rows, primary_stats, secondary_stats


# Creates JSON train/validation/test splits and diagnostics.
def split_and_save(save_dir, dataset_path, train_ratio, val_ratio, seed):
    """Split dataset, write JSON files, and print label bias diagnostics."""

    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoad(dataset_path=dataset_path)
    data = loader.datalabs()

    random.seed(seed)
    random.shuffle(data)

    # The project uses a single shuffle followed by contiguous slicing to
    # produce stable train/validation/test partitions.
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)

    train_data = data[:train_len]
    val_data = data[train_len: train_len + val_len]
    test_data = data[train_len + val_len:]

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "total": data,
    }

    total_size = len(data)
    for name in ["train", "val", "test"]:
        split_data = splits[name]
        path = os.path.join(save_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        size = len(split_data)
        percentage = size / total_size * 100
        print(f"{name.capitalize()}: {size} samples ({percentage:.1f}%) -> {path}")

    compare_and_print_bias_tables(splits)


if __name__ == "__main__":
    split_and_save(
        save_dir="data_splits",
        dataset_path="train_data",
        train_ratio=0.4,
        val_ratio=0.1,
        seed=42,
    )
