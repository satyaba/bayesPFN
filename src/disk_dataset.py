import numpy as np
import pickle
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple


class DiskICLDataset(Dataset):
    """
    Loads pre-generated synthetic datasets from disk for BayesPFN training.

    This eliminates the CPU bottleneck from on-the-fly GBDT-based data generation.
    Instead, datasets are pre-generated once and loaded from disk during training.
    """

    def __init__(
        self,
        data_dir: str,
        n_features: int = 32,
        n_classes: int = 2,
        train_split: float = 0.8,
    ):
        self.data_dir = Path(data_dir)
        self.n_features = n_features
        self.n_classes = n_classes
        self.train_split = train_split

        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        return len(self.metadata["datasets"])

    def __getitem__(self, idx: int) -> dict:
        """
        Load a pre-generated dataset and create train/test split.

        Returns:
            Dictionary with X_train, y_train, X_test, y_test arrays
        """
        dataset_info = self.metadata["datasets"][idx]
        pkl_path = Path(dataset_info["path"])

        if not pkl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pkl_path}")

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {idx}: {e}")

        X = data["X"]
        y = data["y"]

        n_samples = len(y)
        n_train = int(n_samples * self.train_split)

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "n_features": X_train.shape[1],
            "n_train": len(y_train),
            "n_test": len(y_test),
        }


def collate_disk_batch(batch: list) -> dict:
    """
    Collate function for disk-based datasets.

    Converts list of {"X_train", "y_train", "X_test", "y_test"} dicts
    into ICL format tensors compatible with the model.
    """
    max_n_features = max(item["n_features"] for item in batch)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    train_indices_list = []
    test_indices_list = []
    n_train_list = []
    n_test_list = []

    offset = 0
    for item in batch:
        X_train = item["X_train"]
        y_train = item["y_train"]
        X_test = item["X_test"]
        y_test = item["y_test"]

        n_train = item["n_train"]
        n_test = item["n_test"]

        if X_train.shape[1] < max_n_features:
            pad_width = max_n_features - X_train.shape[1]
            X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
            X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')

        n_samples = n_train + n_test
        train_indices = list(range(n_train))
        test_indices = [n_train + i for i in range(n_test)]

        X_all = np.vstack([X_train, X_test])

        X_train_list.append(X_all)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        train_indices_list.append(np.array(train_indices) + offset)
        test_indices_list.append(np.array(test_indices) + offset)
        n_train_list.append(n_train)
        n_test_list.append(n_test)

        offset += n_samples

    return {
        "features": torch.tensor(np.vstack(X_train_list), dtype=torch.float32),
        "train_indices": torch.tensor(np.concatenate(train_indices_list), dtype=torch.long),
        "test_indices": torch.tensor(np.concatenate(test_indices_list), dtype=torch.long),
        "train_labels": torch.tensor(np.concatenate(y_train_list), dtype=torch.long),
        "test_labels": torch.tensor(np.concatenate(y_test_list), dtype=torch.long),
        "n_features_list": [item["n_features"] for item in batch],
    }


if __name__ == "__main__":
    print("DiskICLDataset module loaded successfully")
    print("Usage:")
    print("  1. Generate datasets: python scripts/generate_data.py --n-datasets 500")
    print("  2. Load in training:  from disk_dataset import DiskICLDataset")
