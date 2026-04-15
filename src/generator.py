import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import torch


class SyntheticDataGenerator:
    """
    Generate synthetic tabular datasets for BayesPFN pretraining.

    Uses GBDT as the "ground truth" function to create realistic feature-label
    relationships, similar to how TabPFN uses SCMs and BNNs.

    The imbalance is controlled via StratifiedZoneSampler.
    """

    def __init__(
        self,
        n_features: int = 32,
        n_samples_range: Tuple[int, int] = (500, 2000),
        n_classes: int = 2,
        feature_interaction_degree: int = 2,
    ):
        self.n_features = n_features
        self.n_samples_range = n_samples_range
        self.n_classes = n_classes
        self.feature_interaction_degree = feature_interaction_degree

    def generate_features(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate feature matrix with realistic distributions."""
        X = np.random.randn(n_samples, n_features)

        n_mixuture = np.random.randint(1, 4)
        for _ in range(n_mixuture):
            center = np.random.randn(n_features) * np.random.uniform(1, 5)
            scale = np.random.uniform(0.5, 2.0, size=n_features)
            mixture_weight = np.random.uniform(0.1, 0.3)
            X += mixture_weight * np.random.randn(n_samples, n_features) * scale + center

        X = StandardScaler().fit_transform(X)

        if np.random.random() < 0.3 and n_features >= 3:
            n_interact = np.random.randint(2, min(5, n_features))
            interaction_features = np.random.choice(
                n_features, n_interact, replace=False
            )
            for i in interaction_features:
                for j in interaction_features:
                    if i < j:
                        X[:, i] *= X[:, j]

        return X

    def generate_labels_via_gbdt(self, X: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Generate labels using a GBDT model as ground truth."""
        if self.n_classes == 2:
            y = np.zeros(X.shape[0], dtype=int)
            mask_class1 = np.random.choice(
                [True, False], size=X.shape[0], p=[0.5, 0.5]
            )
            y[mask_class1] = 1
        else:
            y = np.random.randint(0, self.n_classes, size=X.shape[0])

        try:
            clf = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=random_state,
            )
            clf.fit(X, y)
            y = clf.predict(X)
        except Exception:
            pass

        return y

    def apply_imbalance(self, X: np.ndarray, y: np.ndarray, pi: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply class imbalance by subsampling majority class to achieve target π.

        Args:
            X: Feature matrix
            y: Labels
            pi: Target minority proportion (minority / total)

        Returns:
            Imbalanced X, y
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        minority_class = unique_classes[np.argmin(counts)]

        n_minority = counts.min()
        n_majority_target = int(n_minority / pi * (1 - pi))

        mask_majority = y == majority_class
        mask_minority = y == minority_class

        n_majority_current = mask_majority.sum()
        if n_majority_current <= n_majority_target:
            return X, y

        keep_indices = np.where(mask_minority)[0].tolist()
        majority_indices = np.where(mask_majority)[0].tolist()
        keep_majority = np.random.choice(
            majority_indices, size=n_majority_target, replace=False
        )
        keep_indices.extend(keep_majority.tolist())

        keep_indices = np.array(keep_indices)
        np.random.shuffle(keep_indices)

        return X[keep_indices], y[keep_indices]

    def generate_dataset(
        self,
        pi: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single synthetic dataset.

        Args:
            pi: Minority proportion. If None, samples from StratifiedZoneSampler.
            random_state: Random seed for reproducibility.

        Returns:
            X, y as NumPy arrays
        """
        np.random.seed(random_state)

        n_samples = np.random.randint(*self.n_samples_range)
        n_features = self.n_features

        X = self.generate_features(n_samples, n_features)
        y = self.generate_labels_via_gbdt(X, random_state=random_state)

        if pi is None:
            sampler = StratifiedZoneSampler()
            pi = sampler.sample_minority_proportion()

        X_imb, y_imb = self.apply_imbalance(X, y, pi)

        return X_imb, y_imb

    def generate_batch(
        self, n_datasets: int, pi_values: Optional[np.ndarray] = None
    ) -> list:
        """
        Generate a batch of synthetic datasets.

        Args:
            n_datasets: Number of datasets to generate
            pi_values: Pre-sampled minority proportions (for exact control)

        Returns:
            List of (X, y) tuples
        """
        datasets = []
        for i in range(n_datasets):
            pi = pi_values[i] if pi_values is not None else None
            X, y = self.generate_dataset(pi=pi, random_state=42 + i)
            datasets.append((X, y))
        return datasets


class ICLDataset:
    """
    In-context learning dataset wrapper for PFN training.

    Converts (X_train, y_train, X_test) into the flat sequence format
    that PFN transformers expect.
    """

    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes

    def create_icl_sequence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        """
        Create an ICL sequence from train and test data.

        Format:
        - [CLS] token
        - Feature tokens for train samples
        - Label tokens for train samples
        - Feature tokens for test samples

        Returns:
            Dictionary with tensors ready for transformer input
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[1]

        all_features = np.vstack([X_train, X_test])

        train_indices = list(range(n_train))
        test_indices = [n_train + i for i in range(n_test)]

        return {
            "features": all_features,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_labels": y_train,
            "n_features": n_features,
            "n_train": n_train,
            "n_test": n_test,
        }

    def to_tensors(self, icl_seq: dict, device: str = "cuda") -> dict:
        """Convert ICL sequence dictionary to torch tensors."""
        return {
            "features": torch.tensor(icl_seq["features"], dtype=torch.float32, device=device),
            "train_indices": torch.tensor(icl_seq["train_indices"], dtype=torch.long, device=device),
            "test_indices": torch.tensor(icl_seq["test_indices"], dtype=torch.long, device=device),
            "train_labels": torch.tensor(icl_seq["train_labels"], dtype=torch.long, device=device),
            "n_features": icl_seq["n_features"],
            "n_train": icl_seq["n_train"],
            "n_test": icl_seq["n_test"],
        }


if __name__ == "__main__":
    from imbalance import StratifiedZoneSampler

    np.random.seed(42)
    sampler = StratifiedZoneSampler()
    generator = SyntheticDataGenerator()

    pi = sampler.sample_minority_proportion()
    print(f"Sampled π: {pi:.4f} (imbalance ratio: {1/pi - 1:.1f}:1)")

    X, y = generator.generate_dataset(pi=pi)
    print(f"Generated dataset: X shape={X.shape}, y shape={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
