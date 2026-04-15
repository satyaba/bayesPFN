import torch
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from typing import Optional, Tuple, List
import pandas as pd
from pathlib import Path

from model import BayesPFNv1
from generator import ICLDataset


class Evaluator:
    """Evaluation pipeline for BayesPFN models."""

    def __init__(
        self,
        model: BayesPFNv1,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.icl_dataset = ICLDataset()

    def evaluate_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Evaluate on a single train/test split."""
        icl_seq = self.icl_dataset.create_icl_sequence(X_train, y_train, X_test)

        features = torch.tensor(icl_seq["features"], dtype=torch.float32, device=self.device)
        train_indices = torch.tensor(icl_seq["train_indices"], dtype=torch.long, device=self.device)
        test_indices = torch.tensor(icl_seq["test_indices"], dtype=torch.long, device=self.device)
        train_labels = torch.tensor(icl_seq["train_labels"], dtype=torch.long, device=self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(features, train_indices, test_indices, train_labels)
            preds = torch.argmax(probs, dim=-1)

        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        y_test_np = y_test

        n_classes = self.model.n_classes

        metrics = {
            "accuracy": accuracy_score(y_test_np, preds_np),
            "balanced_accuracy": balanced_accuracy_score(y_test_np, preds_np),
            "f1_macro": f1_score(y_test_np, preds_np, average="macro", zero_division=0),
        }

        if n_classes == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test_np, probs_np[:, 1])
            except ValueError:
                metrics["roc_auc"] = None
        else:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_test_np, probs_np, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["roc_auc"] = None

        per_class_metrics = {}
        for c in range(n_classes):
            mask = y_test_np == c
            if mask.sum() > 0:
                per_class_metrics[f"recall_class_{c}"] = (preds_np[mask] == c).sum() / mask.sum()
                per_class_metrics[f"count_class_{c}"] = mask.sum()

        metrics.update(per_class_metrics)

        return metrics

    def evaluate_crossvalidation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Evaluate using cross-validation on a single dataset."""
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        all_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_metrics = self.evaluate_single(X_train, y_train, X_test, y_test)
            fold_metrics["fold"] = fold
            all_metrics.append(fold_metrics)

        df = pd.DataFrame(all_metrics)

        summary = {
            "accuracy_mean": df["accuracy"].mean(),
            "accuracy_std": df["accuracy"].std(),
            "balanced_accuracy_mean": df["balanced_accuracy"].mean(),
            "balanced_accuracy_std": df["balanced_accuracy"].std(),
            "f1_macro_mean": df["f1_macro"].mean(),
            "f1_macro_std": df["f1_macro"].std(),
        }

        if "roc_auc" in df.columns and df["roc_auc"].notna().any():
            summary["roc_auc_mean"] = df["roc_auc"].mean()
            summary["roc_auc_std"] = df["roc_auc"].std()

        return {"summary": summary, "per_fold": all_metrics}

    def evaluate_multiple_datasets(
        self,
        datasets: List[Tuple[str, np.ndarray, np.ndarray]],
        n_splits: int = 5,
    ) -> dict:
        """
        Evaluate on multiple datasets.

        Args:
            datasets: List of (name, X, y) tuples

        Returns:
            Dictionary with results per dataset
        """
        results = {}
        for name, X, y in datasets:
            print(f"Evaluating {name}...")
            cv_results = self.evaluate_crossvalidation(X, y, n_splits=n_splits)
            results[name] = cv_results
            print(f"  Balanced Accuracy: {cv_results['summary']['balanced_accuracy_mean']:.4f} ± {cv_results['summary']['balanced_accuracy_std']:.4f}")
            print(f"  F1-Macro: {cv_results['summary']['f1_macro_mean']:.4f} ± {cv_results['summary']['f1_macro_std']:.4f}")

        return results

    def compute_coverage_gap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_coverage: float = 0.9,
    ) -> dict:
        """
        Compute coverage gap for conformal prediction.

        This is a simplified version since full conformal calibration is Innovation 3.
        """
        icl_seq = self.icl_dataset.create_icl_sequence(X_train, y_train, X_test)

        features = torch.tensor(icl_seq["features"], dtype=torch.float32, device=self.device)
        train_indices = torch.tensor(icl_seq["train_indices"], dtype=torch.long, device=self.device)
        test_indices = torch.tensor(icl_seq["test_indices"], dtype=torch.long, device=self.device)
        train_labels = torch.tensor(icl_seq["train_labels"], dtype=torch.long, device=self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(features, train_indices, test_indices, train_labels)

        n_classes = self.model.n_classes
        coverage_gaps = {}

        for c in range(n_classes):
            mask = y_test == c
            if mask.sum() > 0:
                c_pred_probs = probs[mask, c].cpu().numpy()
                c_actual = (y_test[mask] == c).astype(int)

                empirical_coverage = c_pred_probs.mean()
                gap = abs(empirical_coverage - target_coverage)
                coverage_gaps[f"coverage_gap_class_{c}"] = gap

        max_gap = max(coverage_gaps.values()) if coverage_gaps else 0.0
        coverage_gaps["max_coverage_gap"] = max_gap

        return coverage_gaps


def load_model_from_checkpoint(
    checkpoint_path: str,
    n_features: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    n_classes: int = 2,
    device: str = "cuda",
) -> BayesPFNv1:
    """Load a model from checkpoint."""
    model = BayesPFNv1(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    print("Testing evaluator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from model import BayesPFNv1
    model = BayesPFNv1(n_features=30, d_model=64, n_heads=2, n_layers=2, n_classes=2)
    model.to(device)

    evaluator = Evaluator(model, device=device)

    X, y = load_breast_cancer(return_X_y=True)
    results = evaluator.evaluate_crossvalidation(X, y, n_splits=3)

    print(f"\nBalanced Accuracy: {results['summary']['balanced_accuracy_mean']:.4f} ± {results['summary']['balanced_accuracy_std']:.4f}")
    print(f"F1-Macro: {results['summary']['f1_macro_mean']:.4f} ± {results['summary']['f1_macro_std']:.4f}")

    print("\nEvaluator test passed!")
