import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PFNTransformer(nn.Module):
    """
    Minimal PFN-style transformer for in-context learning on tabular data.

    Architecture:
    - Learnable [CLS] token + feature embeddings
    - 2-4 transformer layers with multi-head self-attention
    - Classification head

    This is a simplified version inspired by TabPFN's architecture,
    designed for continued pretraining on imbalance-stratified data.
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.max_seq_length = max_seq_length

        if d_ff is None:
            d_ff = d_model * 4

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.feature_embedding = nn.Linear(n_features, d_model)
        self.position_embedding = nn.Embedding(max_seq_length + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

        self.classification_head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        features: torch.Tensor,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for ICL prediction.

        Args:
            features: Feature vector [seq_length, n_features]
            train_indices: Indices of training samples in sequence
            test_indices: Indices of test samples in sequence
            train_labels: Labels for training samples (for conditioning)

        Returns:
            logits for test samples [n_test, n_classes]
        """
        seq_length = features.shape[0]
        n_train = len(train_indices)
        n_test = len(test_indices)

        all_indices = torch.cat([train_indices, test_indices])

        x = features[all_indices]

        x = self.feature_embedding(x)

        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        cls_tokens = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        test_positions = torch.arange(1 + n_train, 1 + n_train + n_test, device=x.device)
        test_outputs = x[:, test_positions, :]

<<<<<<< HEAD
        logits = self.classification_head(cls_output)
        logits = logits.expand(n_test, -1)
=======
        logits = self.classification_head(test_outputs.squeeze(0))
>>>>>>> feature/bayespfn-v1-imbalance-prior

        return logits


class BayesPFNv1(nn.Module):
    """
    BayesPFN-v1: Imbalance-stratified PFN with single classification head.

    This is the base model for Innovation 1: imbalance-stratified pretraining.
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_length: int = 15000,
    ):
        super().__init__()

        self.pfn = PFNTransformer(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        self.n_classes = n_classes

    def forward(
        self,
        features: torch.Tensor,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - returns classification logits."""
        return self.pfn(features, train_indices, test_indices, train_labels)

    def predict_proba(
        self,
        features: torch.Tensor,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(features, train_indices, test_indices, train_labels)
        return F.softmax(logits, dim=-1)

    def predict(
        self,
        features: torch.Tensor,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        train_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted class labels."""
        probs = self.predict_proba(features, train_indices, test_indices, train_labels)
        return torch.argmax(probs, dim=-1)


def create_model(
    n_features: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    n_classes: int = 2,
    model_type: str = "bayespfn_v1",
    **kwargs,
) -> nn.Module:
    """Factory function to create models."""
    if model_type == "bayespfn_v1":
        return BayesPFNv1(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            **kwargs,
        )
    elif model_type == "pfn_transformer":
        return PFNTransformer(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    torch.manual_seed(42)

    model = BayesPFNv1(
        n_features=32,
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_classes=2,
    )

    batch_size = 2
    n_train = 100
    n_test = 50
    n_features = 32

    features = torch.randn(n_train + n_test, n_features)
    train_indices = torch.arange(n_train, dtype=torch.long)
    test_indices = torch.arange(n_train, n_train + n_test, dtype=torch.long)
    train_labels = torch.randint(0, 2, size=(n_train,))

    logits = model(features, train_indices, test_indices, train_labels)
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
