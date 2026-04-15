import numpy as np
from typing import Tuple, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.imbalance import StratifiedZoneSampler
from src.generator import SyntheticDataGenerator


def generate_imbalance_stratified_datasets(
    n_datasets: int = 2500,
    n_features: int = 32,
    n_samples_range: Tuple[int, int] = (500, 2000),
    n_classes: int = 2,
    zone_a_ratio: Tuple[float, float] = (1.0, 5.0),
    zone_b_ratio: Tuple[float, float] = (5.0, 10.0),
    zone_c_ratio: Tuple[float, float] = (10.0, 100.0),
    zone_proportions: Tuple[float, float, float] = (0.60, 0.10, 0.30),
    random_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate imbalance-stratified synthetic datasets for head training.

    Each dataset is split into train/test for the ICL learning paradigm.
    The imbalance is controlled via StratifiedZoneSampler.

    Args:
        n_datasets: Number of datasets to generate
        n_features: Number of features per dataset
        n_samples_range: Range of total samples per dataset
        n_classes: Number of classes
        zone_a_ratio: Imbalance ratio range for Zone A (60% of datasets)
        zone_b_ratio: Imbalance ratio range for Zone B (10% of datasets)
        zone_c_ratio: Imbalance ratio range for Zone C (30% of datasets)
        zone_proportions: Proportions for each zone
        random_seed: Random seed for reproducibility

    Returns:
        List of (X_train, y_train, X_test, y_test) tuples
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sampler = StratifiedZoneSampler(
        zone_a_ratio=zone_a_ratio,
        zone_b_ratio=zone_b_ratio,
        zone_c_ratio=zone_c_ratio,
        zone_proportions=zone_proportions,
    )

    data_gen = SyntheticDataGenerator(
        n_features=n_features,
        n_samples_range=n_samples_range,
        n_classes=n_classes,
    )

    datasets = []
    test_size = 0.2

    for i in range(n_datasets):
        pi = sampler.sample_minority_proportion()
        X, y = data_gen.generate_dataset(pi=pi, random_state=42 + i)

        if X.shape[0] < 10:
            continue

        n_test = max(int(X.shape[0] * test_size), 1)
        n_train = X.shape[0] - n_test

        indices = np.random.permutation(X.shape[0])
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        datasets.append((X_train, y_train, X_test, y_test))

    return datasets


def compute_class_weights(
    y: np.ndarray,
    n_classes: int = 2,
) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.

    Args:
        y: Labels
        n_classes: Number of classes

    Returns:
        Class weights array of shape [n_classes]
    """
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    weights = total / (n_classes * counts + 1e-8)
    return weights


def compute_class_weights_from_datasets(
    datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int = 2,
) -> np.ndarray:
    """
    Compute average class weights across all datasets.

    Args:
        datasets: List of (X_train, y_train, X_test, y_test) tuples
        n_classes: Number of classes

    Returns:
        Average class weights [n_classes]
    """
    all_weights = []
    for X_train, y_train, X_test, y_test in datasets:
        weights = compute_class_weights(y_train, n_classes)
        all_weights.append(weights)

    return np.mean(all_weights, axis=0)
