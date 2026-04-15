#!/usr/bin/env python
"""
Evaluation script for TabPFN Base + Fine-tune Head.

Evaluates the trained head on benchmark datasets (Creditcard, Mammography, etc.)
using TabPFN for feature extraction and the trained head for classification.

Usage:
    python scripts/evaluate_head.py --checkpoint checkpoints/tabpfn_head.pt
    python scripts/evaluate_head.py --benchmark creditcard
    python scripts/evaluate_head.py --benchmark mammography
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tabpfn_head import (
    TabPFNFeatureExtractor,
    TabPFNClassificationHead,
)


BENCHMARK_DATASETS = {
    "creditcard_fraud": {
        "openml_id": 1597,
        "name": "creditcard_fraud",
    },
    "mammography": {
        "openml_id": 43893,
        "name": "mammography",
    },
    "yeast": {
        "openml_id": 181,
        "name": "yeast",
    },
}


def load_benchmark_dataset(benchmark_name: str):
    """Load a benchmark dataset from OpenML."""
    if benchmark_name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_DATASETS.keys())}")

    config = BENCHMARK_DATASETS[benchmark_name]
    openml_id = config["openml_id"]

    print(f"Loading {benchmark_name} from OpenML (ID: {openml_id})...")
    dataset = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    if hasattr(X, "iloc"):
        X = X.values
    if hasattr(y, "iloc"):
        y = y.values

    print(f"  Loaded: X shape={X.shape}, y shape={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")

    return X, y


def evaluate_head_cv(
    extractor: TabPFNFeatureExtractor,
    head: TabPFNClassificationHead,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    device: str = "cuda",
) -> dict:
    """
    Evaluate head using cross-validation.

    Args:
        extractor: TabPFNFeatureExtractor
        head: Trained TabPFNClassificationHead
        X: Features
        y: Labels
        n_splits: Number of CV splits
        device: Device to use

    Returns:
        Dictionary with metrics summary
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_metrics = []
    head.to(device)
    head.eval()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{n_splits}...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        extractor.fit(X_train, y_train)
        embeddings = extractor.extract(X_train, y_train, X_test)
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = head(embeddings)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        y_test_np = y_test.astype(int)
        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_test_np, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test_np, preds),
            "f1_macro": f1_score(y_test_np, preds, average="macro", zero_division=0),
        }
        all_metrics.append(metrics)
        print(f"Bal Acc: {metrics['balanced_accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")

    summary = {
        "accuracy_mean": np.mean([m["accuracy"] for m in all_metrics]),
        "accuracy_std": np.std([m["accuracy"] for m in all_metrics]),
        "balanced_accuracy_mean": np.mean([m["balanced_accuracy"] for m in all_metrics]),
        "balanced_accuracy_std": np.std([m["balanced_accuracy"] for m in all_metrics]),
        "f1_macro_mean": np.mean([m["f1_macro"] for m in all_metrics]),
        "f1_macro_std": np.std([m["f1_macro"] for m in all_metrics]),
    }

    return {"summary": summary, "per_fold": all_metrics}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TabPFN head on benchmarks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tabpfn_head.pt",
        help="Path to head checkpoint",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["creditcard_fraud", "mammography", "yeast", "all"],
        default="all",
        help="Benchmark dataset to evaluate on",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV splits",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TabPFN Base + Fine-tune Head Evaluation")
    print("=" * 60)

    print(f"\n[1/3] Loading head from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    head_config = checkpoint.get("head_config", {"d_model": 128, "n_classes": 2})
    
    head = TabPFNClassificationHead(
        d_model=head_config["d_model"],
        n_classes=head_config["n_classes"],
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(args.device)
    head.eval()
    print(f"  Loaded head: d_model={head_config['d_model']}, n_classes={head_config['n_classes']}")

    print(f"\n[2/3] Initializing TabPFN extractor...")
    extractor = TabPFNFeatureExtractor(device=args.device)

    print(f"\n[3/3] Evaluating on benchmarks...")
    
    if args.benchmark == "all":
        benchmarks_to_run = list(BENCHMARK_DATASETS.keys())
    else:
        benchmarks_to_run = [args.benchmark]

    results = {}
    for benchmark_name in benchmarks_to_run:
        print(f"\n{'=' * 40}")
        print(f"Benchmark: {benchmark_name}")
        print(f"{'=' * 40}")
        
        try:
            X, y = load_benchmark_dataset(benchmark_name)
            
            cv_results = evaluate_head_cv(
                extractor=extractor,
                head=head,
                X=X,
                y=y,
                n_splits=args.n_splits,
                device=args.device,
            )

            results[benchmark_name] = cv_results

            print(f"\nSummary:")
            print(f"  Balanced Accuracy: {cv_results['summary']['balanced_accuracy_mean']:.4f} ± {cv_results['summary']['balanced_accuracy_std']:.4f}")
            print(f"  F1-Macro: {cv_results['summary']['f1_macro_mean']:.4f} ± {cv_results['summary']['f1_macro_std']:.4f}")

        except Exception as e:
            print(f"  Error evaluating {benchmark_name}: {e}")
            results[benchmark_name] = {"error": str(e)}

    print(f"\n{'=' * 60}")
    print("Final Results Summary")
    print(f"{'=' * 60}")
    
    for benchmark_name, cv_results in results.items():
        if "error" in cv_results:
            print(f"{benchmark_name}: ERROR - {cv_results['error']}")
        else:
            print(f"{benchmark_name}:")
            print(f"  Balanced Accuracy: {cv_results['summary']['balanced_accuracy_mean']:.4f} ± {cv_results['summary']['balanced_accuracy_std']:.4f}")
            print(f"  F1-Macro: {cv_results['summary']['f1_macro_mean']:.4f} ± {cv_results['summary']['f1_macro_std']:.4f}")


if __name__ == "__main__":
    main()
