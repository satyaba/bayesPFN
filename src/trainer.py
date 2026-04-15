import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb
import time
from pathlib import Path

from model import BayesPFNv1, create_model
from generator import SyntheticDataGenerator, ICLDataset
from imbalance import StratifiedZoneSampler


class ICLBatchDataset(Dataset):
    """Dataset that yields ICL batches for training."""

    def __init__(
        self,
        generator: SyntheticDataGenerator,
        sampler: StratifiedZoneSampler,
        n_datasets: int,
        n_classes: int = 2,
    ):
        self.generator = generator
        self.sampler = sampler
        self.n_datasets = n_datasets
        self.n_classes = n_classes
        self.icl_dataset = ICLDataset(n_classes=n_classes)

        self.pi_values = sampler.sample_batch(n_datasets)

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx: int) -> dict:
        pi = self.pi_values[idx]
        X, y = self.generator.generate_dataset(pi=pi, random_state=42 + idx)

        n_train = int(len(y) * 0.8)
        indices = np.random.permutation(len(y))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        icl_seq = self.icl_dataset.create_icl_sequence(X_train, y_train, X_test)

        return {
            "features": torch.tensor(icl_seq["features"], dtype=torch.float32),
            "train_indices": torch.tensor(icl_seq["train_indices"], dtype=torch.long),
            "test_indices": torch.tensor(icl_seq["test_indices"], dtype=torch.long),
            "train_labels": torch.tensor(icl_seq["train_labels"], dtype=torch.long),
            "test_labels": torch.tensor(y_test, dtype=torch.long),
            "n_features": icl_seq["n_features"],
        }


def collate_icl_batch(batch: list) -> dict:
    """Collate function for ICL batches."""
    max_n_features = max(item["n_features"] for item in batch)

    features_list = []
    train_indices_list = []
    test_indices_list = []
    train_labels_list = []
    test_labels_list = []
    n_features_list = []
    offset = 0

    for item in batch:
        n_feat = item["n_features"]
        n_train = len(item["train_indices"])
        n_test = len(item["test_indices"])

        features_list.append(item["features"])
        train_indices_list.append(item["train_indices"] + offset)
        test_indices_list.append(item["test_indices"] + offset)
        train_labels_list.append(item["train_labels"])
        test_labels_list.append(item["test_labels"])
        n_features_list.append(item["n_features"])

        offset += n_train + n_test

    return {
        "features": torch.cat(features_list, dim=0),
        "train_indices": torch.cat(train_indices_list, dim=0),
        "test_indices": torch.cat(test_indices_list, dim=0),
        "train_labels": torch.cat(train_labels_list, dim=0),
        "test_labels": torch.cat(test_labels_list, dim=0),
        "n_features_list": n_features_list,
    }


class Trainer:
    """Training loop for BayesPFN-v1."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        log_interval: int = 10,
        save_dir: str = "./checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()

        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        features = batch["features"].to(self.device)
        train_indices = batch["train_indices"].to(self.device)
        test_indices = batch["test_indices"].to(self.device)
        train_labels = batch["train_labels"].to(self.device)
        test_labels = batch["test_labels"].to(self.device)

        logits = self.model(features, train_indices, test_indices, train_labels)

        loss = self.criterion(logits, test_labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == test_labels).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_fn=None,
    ) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0

        for batch in train_loader:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_interval == 0 and log_fn:
                log_fn(metrics)

        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches

        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def save_checkpoint(self, filepath: str, metrics: Optional[dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "metrics": metrics or {},
            "elapsed_time": time.time() - self.start_time,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {filepath}")


def create_training_setup(
    n_features: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    n_classes: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    device: str = "cuda",
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create model, optimizer, and scheduler."""
    model = create_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
        model_type="bayespfn_v1",
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,
        T_mult=2,
    )

    return model, optimizer, scheduler


if __name__ == "__main__":
    print("Testing training setup...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, optimizer, scheduler = create_training_setup(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    generator = SyntheticDataGenerator(n_features_range=(10, 50))
    sampler = StratifiedZoneSampler()
    dataset = ICLBatchDataset(
        generator=generator,
        sampler=sampler,
        n_datasets=100,
    )
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_icl_batch)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    for batch in loader:
        print("Batch features shape:", batch["features"].shape)
        print("Batch test_labels shape:", batch["test_labels"].shape)
        break

    print("Training setup test passed!")
