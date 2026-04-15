#!/usr/bin/env python
"""
Training script for TabPFN Base + Fine-tune Head.

This script:
1. Loads TabPFN as a frozen feature extractor
2. Generates imbalance-stratified synthetic datasets
3. Trains only a linear classification head
4. Saves the trained head checkpoint

Usage:
    python scripts/train_head.py --config configs/tabpfn_head.yaml
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tabpfn_head import (
    TabPFNFeatureExtractor,
    TabPFNClassificationHead,
    generate_imbalance_stratified_datasets,
    HeadTrainer,
    train_head,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TabPFN classification head on synthetic data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tabpfn_head.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tabpfn_head.pt",
        help="Path to save head checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    config = load_config(args.config)
    tabpfn_head_config = config.get("tabpfn_head", config)

    base_config = tabpfn_head_config.get("base", {})
    head_config = tabpfn_head_config.get("head", {})
    training_config = tabpfn_head_config.get("training", {})
    imbalance_config = tabpfn_head_config.get("imbalance", {})

    print("=" * 60)
    print("TabPFN Base + Fine-tune Head Training")
    print("=" * 60)

    print("\n[1/5] Initializing TabPFN feature extractor...")
    extractor = TabPFNFeatureExtractor(
        model_path=base_config.get("model_path", "auto"),
        n_estimators=base_config.get("n_estimators", 8),
        device=args.device,
    )
    print(f"  Model path: {base_config.get('model_path', 'auto')}")
    print(f"  N estimators: {base_config.get('n_estimators', 8)}")
    print(f"  Device: {args.device}")

    print("\n[2/5] Initializing classification head...")
    d_model = head_config.get("d_model", 128)
    n_classes = head_config.get("n_classes", 2)
    head = TabPFNClassificationHead(d_model=d_model, n_classes=n_classes)
    print(f"  d_model: {d_model}")
    print(f"  n_classes: {n_classes}")

    print("\n[3/5] Generating imbalance-stratified synthetic datasets...")
    n_datasets = training_config.get("n_datasets", 2500)
    n_features = training_config.get("n_features", 32)
    datasets = generate_imbalance_stratified_datasets(
        n_datasets=n_datasets,
        n_features=n_features,
        n_samples_range=tuple(training_config.get("n_samples_range", [500, 2000])),
        n_classes=n_classes,
        zone_a_ratio=tuple(imbalance_config.get("zone_a_ratio", [1.0, 5.0])),
        zone_b_ratio=tuple(imbalance_config.get("zone_b_ratio", [5.0, 10.0])),
        zone_c_ratio=tuple(imbalance_config.get("zone_c_ratio", [10.0, 100.0])),
        zone_proportions=tuple(imbalance_config.get("zone_proportions", [0.60, 0.10, 0.30])),
    )
    print(f"  Generated {len(datasets)} datasets")
    print(f"  Features per dataset: {n_features}")
    print(f"  Zone A (r < 5:1): 60%, Zone B (5:1 ≤ r < 10:1): 10%, Zone C (r ≥ 10:1): 30%")

    print("\n[4/5] Training classification head...")
    lr = training_config.get("learning_rate", 1e-3)
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batch_size", 1)

    head, final_metrics = train_head(
        feature_extractor=extractor,
        head=head,
        datasets=datasets,
        epochs=epochs,
        lr=lr,
        device=args.device,
        verbose=True,
    )

    print(f"\n  Final metrics:")
    print(f"    Loss: {final_metrics['loss']:.4f}")
    print(f"    Accuracy: {final_metrics['accuracy']:.4f}")

    print("\n[5/5] Saving checkpoint...")
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "head_config": {
                "d_model": head.d_model,
                "n_classes": head.n_classes,
            },
            "training_config": training_config,
        },
        args.checkpoint,
    )
    print(f"  Saved to: {args.checkpoint}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
