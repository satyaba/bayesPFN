import torch
import torch.nn as nn
from typing import Optional


class TabPFNClassificationHead(nn.Module):
    """
    Simple linear classification head for TabPFN embeddings.

    Architecture:
        Linear(d_model, n_classes)

    This head is trained while keeping the TabPFN base frozen.
    Uses CrossEntropy loss with optional class weights for imbalance.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_classes: int = 2,
    ):
        """
        Initialize classification head.

        Args:
            d_model: Input embedding dimension (TabPFN default: 128)
            n_classes: Number of output classes
        """
        super().__init__()
        self.head = nn.Linear(d_model, n_classes)
        self.d_model = d_model
        self.n_classes = n_classes

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.

        Args:
            embeddings: Input embeddings [..., d_model]

        Returns:
            Class logits [..., n_classes]
        """
        return self.head(embeddings)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            embeddings: Input embeddings [..., d_model]

        Returns:
            Class probabilities [..., n_classes]
        """
        logits = self.forward(embeddings)
        return torch.softmax(logits, dim=-1)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.

        Args:
            embeddings: Input embeddings [..., d_model]

        Returns:
            Class predictions [...]
        """
        proba = self.predict_proba(embeddings)
        return torch.argmax(proba, dim=-1)


def create_head_with_weights(
    checkpoint_path: str,
    d_model: int = 128,
    n_classes: int = 2,
) -> TabPFNClassificationHead:
    """
    Create a classification head and load weights from checkpoint.

    Args:
        checkpoint_path: Path to saved head weights
        d_model: Embedding dimension
        n_classes: Number of classes

    Returns:
        Loaded TabPFNClassificationHead
    """
    head = TabPFNClassificationHead(d_model=d_model, n_classes=n_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(state_dict)
    return head
