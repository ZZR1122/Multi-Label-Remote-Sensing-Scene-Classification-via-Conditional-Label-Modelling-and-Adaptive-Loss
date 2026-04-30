import os
from pathlib import Path
import pandas as pd


# Loads records from image folders and label CSV files.
def load_dataset(images_dir, labels_dir):
    """Build the project record format from image folders and per-class CSV labels."""

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    data = []
    for image_folder in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        primary_label = image_folder.name
        label_file = labels_dir / f"{primary_label}.csv"

        df = pd.read_csv(label_file, index_col=0)
        label_map = df.to_dict(orient="index")

        for image_path in sorted(path for path in image_folder.iterdir() if path.is_file()):
            img_name = image_path.name
            # Each record keeps the original file path, the folder-based primary label,
            # and the row from the corresponding secondary-label CSV.
            if img_name in label_map:
                data.append({
                    "image_name": img_name,
                    "image_path": str(image_path),
                    "primary_label": primary_label,
                    "secondary_labels": label_map[img_name],
                })
    return data


# Provides dataset-loading helpers for downstream scripts.
class DataLoad:
    """Thin wrapper used by downstream scripts to load the prepared dataset."""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def dataset(self):
        """Return raw records with `primary_label` and `secondary_labels`."""

        return load_dataset(
            os.path.join(self.dataset_path, "Images"),
            os.path.join(self.dataset_path, "labels"),
        )

    def datalabs(self):
        """Attach a deterministic primary-label one-hot dictionary to each record."""

        data = self.dataset()

        primary_label_list = sorted(set(item["primary_label"] for item in data))

        for i in range(len(data)):
            primary_label = data[i]["primary_label"]
            # All records share the same class order so train/val/test use identical indices.
            data[i]["primary_label_one_hot"] = encode_primary_label(primary_label, primary_label_list)

        return data


# Normalizes label text for consistent comparisons.
def _normalize_label(s):
    """Normalize label text so display code can suppress duplicate semantics."""

    return " ".join(str(s).lower().strip().replace("_", " ").split())


# Converts a primary label into one-hot dictionary form.
def encode_primary_label(primary_label, all_labels):
    """Encode a primary label into the project's dict-based one-hot format."""

    return {label: (1 if label == primary_label else 0) for label in all_labels}


if __name__ == "__main__":
    dl = DataLoad(dataset_path="train_data")
    dataset = dl.datalabs()

    for item in dataset[:1]:
        print(item)
