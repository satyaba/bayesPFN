#!/usr/bin/env python3
"""Script to generate synthetic datasets for BayesPFN pretraining."""

import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import SyntheticDataGenerator
from imbalance import StratifiedZoneSampler, verify_zone_properties


def generate_dataset_cache(
    output_dir: Path,
    n_datasets: int,
    n_features: int = 32,
    n_samples_range: tuple = (500, 2000),
    n_classes: int = 2,
    verify: bool = False,
) -> dict:
    """
    Generate a cache of synthetic datasets.

    Returns a dictionary with metadata about the generated datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = StratifiedZoneSampler()
    generator = SyntheticDataGenerator(
        n_features=n_features,
        n_samples_range=n_samples_range,
        n_classes=n_classes,
    )

    if verify:
        print("Verifying zone sampler properties...")
        verify_zone_properties(sampler)
        print()

    pi_values = sampler.sample_batch(n_datasets)

    metadata = {
        "n_datasets": n_datasets,
        "n_features": n_features,
        "n_samples_range": list(n_samples_range),
        "n_classes": n_classes,
        "pi_values": pi_values.tolist(),
        "datasets": [],
    }

    print(f"Generating {n_datasets} datasets...")
    for i in tqdm(range(n_datasets)):
        X, y = generator.generate_dataset(pi=pi_values[i], random_state=42 + i)

        dataset_path = output_dir / f"dataset_{i:06d}.pkl"
        with open(dataset_path, "wb") as f:
            pickle.dump({"X": X, "y": y, "pi": pi_values[i]}, f)

        metadata["datasets"].append({
            "index": i,
            "path": str(dataset_path),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "pi": float(pi_values[i]),
            "imbalance_ratio": float((1 - pi_values[i]) / pi_values[i]),
        })

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    imbalance_stats = compute_imbalance_stats(metadata["datasets"])
    print(f"\nImbalance distribution:")
    print(f"  Zone A (r < 5:1):   {imbalance_stats['zone_a_pct']:.1f}%")
    print(f"  Zone B (5:1 ≤ r < 10:1): {imbalance_stats['zone_b_pct']:.1f}%")
    print(f"  Zone C (r ≥ 10:1):  {imbalance_stats['zone_c_pct']:.1f}%")

    return metadata


def compute_imbalance_stats(datasets: list) -> dict:
    """Compute statistics about imbalance distribution."""
    ratios = [d["imbalance_ratio"] for d in datasets]

    zone_a = sum(1 for r in ratios if r < 5)
    zone_b = sum(1 for r in ratios if 5 <= r < 10)
    zone_c = sum(1 for r in ratios if r >= 10)

    n = len(ratios)
    return {
        "zone_a_pct": zone_a / n * 100,
        "zone_b_pct": zone_b / n * 100,
        "zone_c_pct": zone_c / n * 100,
        "mean_ratio": np.mean(ratios),
        "median_ratio": np.median(ratios),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for BayesPFN")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/synthetic",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=500,
        help="Number of datasets to generate",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=32,
        help="Number of features per dataset",
    )
    parser.add_argument(
        "--n-samples-min",
        type=int,
        default=500,
        help="Minimum number of samples per dataset",
    )
    parser.add_argument(
        "--n-samples-max",
        type=int,
        default=2000,
        help="Maximum number of samples per dataset",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify zone sampler properties before generating",
    )

    args = parser.parse_args()

    generate_dataset_cache(
        output_dir=args.output_dir,
        n_datasets=args.n_datasets,
        n_features=args.n_features,
        n_samples_range=(args.n_samples_min, args.n_samples_max),
        verify=args.verify,
    )

    print(f"\nGenerated datasets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
