import numpy as np
import torch
from typing import Tuple, Optional
from tabpfn import TabPFNClassifier


class TabPFNFeatureExtractor:
    """
    TabPFN wrapper for embedding extraction.

    Uses TabPFN's pretrained transformer as a frozen feature extractor.
    Provides per-sample embeddings from the transformer's output.
    """

    def __init__(
        self,
        model_path: str = "auto",
        n_estimators: int = 8,
        device: Optional[str] = None,
    ):
        """
        Initialize TabPFN feature extractor.

        Args:
            model_path: Path to TabPFN checkpoint or "auto" for download
            n_estimators: Number of estimators in the TabPFN ensemble
            device: Device to use ("cuda" or "cpu"), auto-detects if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tabpfn = TabPFNClassifier(
            model_path=model_path,
            n_estimators=n_estimators,
            device=device,
        )
        self.device = device
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabPFNFeatureExtractor":
        """
        Fit TabPFN on training data (required before embedding extraction).

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training labels [n_train]

        Returns:
            self
        """
        self.tabpfn.fit(X_train, y_train)
        self._fitted = True
        return self

    def extract(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        pool_estimators: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings from TabPFN for test samples.

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training labels [n_train]
            X_test: Test features [n_test, n_features]
            pool_estimators: If True, mean-pool across estimators
                           If False, returns [n_estimators, n_test, d_model]

        Returns:
            Embeddings array:
            - If pool_estimators=True: [n_test, d_model]
            - If pool_estimators=False: [n_estimators, n_test, d_model]
        """
        if not self._fitted:
            raise RuntimeError("TabPFN not fitted. Call fit() first.")

        embeddings = self.tabpfn.get_embeddings(X_test, data_source="test")

        if pool_estimators:
            embeddings = np.mean(embeddings, axis=0)

        return embeddings

    def extract_from_fitted(
        self,
        X_test: np.ndarray,
        pool_estimators: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings assuming TabPFN is already fitted.

        This is faster when the model is already fitted and you only
        want to extract embeddings for new test data.

        Args:
            X_test: Test features [n_test, n_features]
            pool_estimators: If True, mean-pool across estimators

        Returns:
            Embeddings array
        """
        if not self._fitted:
            raise RuntimeError("TabPFN not fitted. Call fit() first.")

        embeddings = self.tabpfn.get_embeddings(X_test, data_source="test")

        if pool_estimators:
            embeddings = np.mean(embeddings, axis=0)

        return embeddings

    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension d_model.

        Returns:
            Embedding dimension (typically 128 for TabPFN v2)
        """
        return 128

    def to(self, device: str) -> "TabPFNFeatureExtractor":
        """Move model to specified device."""
        self.device = device
        self.tabpfn.to(device)
        return self


def extract_features_batch(
    extractor: TabPFNFeatureExtractor,
    datasets: list,
    pool_estimators: bool = True,
) -> list:
    """
    Extract embeddings from multiple datasets.

    Args:
        extractor: Fitted TabPFNFeatureExtractor
        datasets: List of (X_train, y_train, X_test, y_test) tuples
        pool_estimators: Whether to mean-pool across estimators

    Returns:
        List of (embeddings, y_test) tuples
    """
    results = []
    for X_train, y_train, X_test, y_test in datasets:
        embeddings = extractor.extract(
            X_train, y_train, X_test, pool_estimators=pool_estimators
        )
        results.append((embeddings, y_test))
    return results
