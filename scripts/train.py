#!/usr/bin/env python3
"""Training script for BayesPFN-v1."""

import argparse
import sys
from pathlib import Path
import torch
import wandb
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import BayesPFNv1, create_model
from trainer import Trainer, ICLBatchDataset, collate_icl_batch, create_training_setup
from generator import SyntheticDataGenerator
from imbalance import StratifiedZoneSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train BayesPFN-v1")

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "bayespfn_v1.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=None,
        help="Number of datasets to train on (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bayespfn",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    config = load_config(args.config)
    model_config = config["bayespfn_v1"]["model"]
    train_config = config["bayespfn_v1"]["training"]
    data_config = config["bayespfn_v1"]["data"]

    n_datasets = args.n_datasets if args.n_datasets else data_config["scaling"]["sanity_check"]

    print("=" * 60)
    print("BayesPFN-v1 Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Datasets: {n_datasets}")
    print(f"Device: {args.device}")
    print()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**model_config, **train_config},
    )

    model, optimizer, scheduler = create_training_setup(
        n_features=model_config["n_features"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        n_classes=model_config["n_classes"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        device=args.device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    sampler = StratifiedZoneSampler(
        zone_a_ratio=tuple(data_config["imbalance"]["zone_a_ratio"]),
        zone_b_ratio=tuple(data_config["imbalance"]["zone_b_ratio"]),
        zone_c_ratio=tuple(data_config["imbalance"]["zone_c_ratio"]),
        zone_proportions=tuple(data_config["imbalance"]["zone_proportions"]),
        power_law_exponent=data_config["imbalance"]["power_law_exponent"],
    )

    generator = SyntheticDataGenerator(
        n_features=data_config["n_features"],
        n_samples_range=tuple(data_config["n_samples_range"]),
        n_classes=data_config["n_classes"],
    )

    dataset = ICLBatchDataset(
        generator=generator,
        sampler=sampler,
        n_datasets=n_datasets,
        n_classes=model_config["n_classes"],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        collate_fn=collate_icl_batch,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_interval=train_config["log_interval"],
        save_dir=str(checkpoint_dir),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"Training on {n_datasets} datasets...")
    print(f"Batches per epoch: {len(loader)}")
    print()

    for epoch in range(train_config["num_epochs"]):
        trainer.epoch = epoch

        def log_fn(metrics):
            metrics["epoch"] = epoch
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(metrics)

        results = trainer.train_epoch(loader, log_fn=log_fn)

        print(
            f"Epoch {epoch}: loss={results['loss']:.4f}, "
            f"accuracy={results['accuracy']:.4f}"
        )

        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f"bayespfn_v1_epoch{epoch}.ckpt"
            trainer.save_checkpoint(
                str(checkpoint_path),
                metrics={"epoch": epoch, **results},
            )

    final_checkpoint_path = checkpoint_dir / "bayespfn_v1_final.ckpt"
    trainer.save_checkpoint(str(final_checkpoint_path), metrics=results)

    wandb.finish()

    print("\nTraining complete!")
    print(f"Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
