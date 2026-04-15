#!/usr/bin/env python3
"""Evaluation script for BayesPFN-v1 on imbalanced benchmarks."""

import argparse
import sys
from pathlib import Path
import torch
import yaml
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import BayesPFNv1
from evaluation import Evaluator, load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BayesPFN-v1")

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "bayespfn_v1.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["creditcard", "mammography", "yeast", "all"],
        default="all",
        help="Which dataset to evaluate on",
    )

    return parser.parse_args()


def load_openml_dataset(openml_id: int) -> tuple:
    """Load dataset from OpenML by ID."""
    try:
        import openml
        dataset = openml.datasets.get_dataset(openml_id)
        X, y, _, _ = dataset.get_data()
        return X.values, y.values
    except ImportError:
        print("openml not installed. Using synthetic fallback.")
        return None, None


def get_benchmark_datasets():
    """Get benchmark datasets for imbalance evaluation."""
    return [
        {"name": "creditcard_fraud", "openml_id": 1597},
        {"name": "mammography", "openml_id": 43893},
        {"name": "yeast", "openml_id": 181},
    ]


def main():
    args = parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    model_config = config["bayespfn_v1"]["model"]

    print("=" * 60)
    print("BayesPFN-v1 Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    print("Loading model...")
    model = load_model_from_checkpoint(
        args.checkpoint,
        n_features=model_config["n_features"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        n_classes=model_config["n_classes"],
        device=args.device,
    )

    evaluator = Evaluator(model, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        benchmark_datasets = get_benchmark_datasets()
    else:
        name_map = {
            "creditcard": "creditcard_fraud",
            "mammography": "mammography",
            "yeast": "yeast",
        }
        benchmark_datasets = [
            {"name": name_map[args.dataset], "openml_id": None}
        ]

    all_results = {}
    for bench_config in benchmark_datasets:
        name = bench_config["name"]
        openml_id = bench_config["openml_id"]

        print(f"\n{'=' * 40}")
        print(f"Evaluating on {name}")
        print(f"{'=' * 40}")

        if openml_id:
            print(f"Loading OpenML dataset {openml_id}...")
            X, y = load_openml_dataset(openml_id)
        else:
            print("No OpenML ID provided, skipping.")
            continue

        if X is None:
            print("Failed to load dataset, skipping.")
            continue

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        imbalance_ratio = np.max(np.bincount(y)) / np.min(np.bincount(y))
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

        print(f"\nRunning {args.n_splits}-fold cross-validation...")
        results = evaluator.evaluate_crossvalidation(
            X, y,
            n_splits=args.n_splits,
        )

        print(f"\nResults for {name}:")
        print(f"  Balanced Accuracy: {results['summary']['balanced_accuracy_mean']:.4f} ± {results['summary']['balanced_accuracy_std']:.4f}")
        print(f"  F1-Macro: {results['summary']['f1_macro_mean']:.4f} ± {results['summary']['f1_macro_std']:.4f}")
        if "roc_auc_mean" in results["summary"]:
            print(f"  ROC-AUC: {results['summary']['roc_auc_mean']:.4f} ± {results['summary']['roc_auc_std']:.4f}")

        all_results[name] = results

        results_df = pd.DataFrame(results["per_fold"])
        results_path = output_dir / f"{name}_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"  Per-fold results saved to {results_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    summary_data = []
    for name, results in all_results.items():
        summary_data.append({
            "dataset": name,
            "balanced_accuracy": results["summary"]["balanced_accuracy_mean"],
            "balanced_accuracy_std": results["summary"]["balanced_accuracy_std"],
            "f1_macro": results["summary"]["f1_macro_mean"],
            "f1_macro_std": results["summary"]["f1_macro_std"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
