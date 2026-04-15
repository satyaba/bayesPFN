import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm


class HeadTrainer:
    """
    Trainer for the TabPFN classification head.

    Freezes TabPFN base and only trains the linear classification head.
    Uses CrossEntropy loss with class weights for imbalance handling.
    """

    def __init__(
        self,
        feature_extractor,
        head: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        """
        Initialize head trainer.

        Args:
            feature_extractor: TabPFNFeatureExtractor (frozen)
            head: TabPFNClassificationHead (trainable)
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for AdamW
        """
        self.extractor = feature_extractor
        self.head = head.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        y_true: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            embeddings: TabPFN embeddings [batch, d_model]
            y_true: True labels [batch]
            class_weights: Class weights for imbalance [n_classes]

        Returns:
            Loss scalar
        """
        logits = self.head(embeddings)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion(logits, y_true)

    def train_step(
        self,
        embeddings: torch.Tensor,
        y_true: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            embeddings: TabPFN embeddings [batch, d_model]
            y_true: True labels [batch]
            class_weights: Class weights for imbalance

        Returns:
            Dictionary of metrics
        """
        self.optimizer.zero_grad()

        embeddings = embeddings.to(self.device)
        y_true = y_true.to(self.device)

        loss = self.compute_loss(embeddings, y_true, class_weights)

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = self.head.predict(embeddings)
            accuracy = (preds == y_true).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

    def train_epoch(
        self,
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        class_weights: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Train on a single dataset (one epoch).

        Args:
            dataset: (X_train, y_train, X_test, y_test)
            class_weights: Class weights for imbalance
            verbose: Whether to print progress

        Returns:
            Dictionary of averaged metrics
        """
        X_train, y_train, X_test, y_test = dataset

        embeddings = self.extractor.extract(X_train, y_train, X_test)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        y_true = torch.tensor(y_test, dtype=torch.long)

        metrics = self.train_step(embeddings, y_true, class_weights)

        if verbose:
            print(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def train_batch(
        self,
        datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        class_weights: Optional[torch.Tensor] = None,
        epochs: int = 1,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Train on a batch of datasets.

        Args:
            datasets: List of (X_train, y_train, X_test, y_test) tuples
            class_weights: Class weights for imbalance
            epochs: Number of epochs per dataset
            verbose: Whether to print progress

        Returns:
            Dictionary of averaged metrics
        """
        total_metrics = {"loss": 0.0, "accuracy": 0.0}
        n_steps = 0

        iterator = tqdm(datasets, desc="Training head") if verbose else datasets

        for dataset in iterator:
            for _ in range(epochs):
                metrics = self.train_epoch(dataset, class_weights, verbose=False)
                total_metrics["loss"] += metrics["loss"]
                total_metrics["accuracy"] += metrics["accuracy"]
                n_steps += 1

                if verbose:
                    iterator.set_postfix({
                        "loss": metrics["loss"],
                        "acc": metrics["accuracy"],
                    })

        return {
            "loss": total_metrics["loss"] / n_steps,
            "accuracy": total_metrics["accuracy"] / n_steps,
        }

    def save_checkpoint(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save head checkpoint.

        Args:
            filepath: Path to save checkpoint
            metadata: Optional metadata dict
        """
        checkpoint = {
            "head_state_dict": self.head.state_dict(),
            "head_config": {
                "d_model": self.head.d_model,
                "n_classes": self.head.n_classes,
            },
        }
        if metadata is not None:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load head checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.head.load_state_dict(checkpoint["head_state_dict"])


def train_head(
    feature_extractor,
    head: nn.Module,
    datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    class_weights: Optional[np.ndarray] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Convenience function to train the head.

    Args:
        feature_extractor: TabPFNFeatureExtractor (frozen)
        head: TabPFNClassificationHead (trainable)
        datasets: List of (X_train, y_train, X_test, y_test)
        class_weights: Class weights for imbalance
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        verbose: Whether to print progress

    Returns:
        Tuple of (trained head, final metrics)
    """
    trainer = HeadTrainer(
        feature_extractor=feature_extractor,
        head=head,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
    )

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    final_metrics = trainer.train_batch(
        datasets=datasets,
        class_weights=class_weights,
        epochs=epochs,
        verbose=verbose,
    )

    return head, final_metrics
