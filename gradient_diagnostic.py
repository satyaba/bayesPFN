#!/usr/bin/env python3
"""Gradient Flow Diagnostic for BayesPFN.

This script identifies where gradients are blocked or zero in the model.
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from torch import nn
from disk_dataset import DiskICLDataset, collate_disk_batch
from torch.utils.data import DataLoader
from model import BayesPFNv1


class GradientChecker:
    """Check gradient flow through model layers."""

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.grad_norms = {}

    def check_gradients(self, batch, check_frequency=100):
        """Run forward/backward and check gradients at each layer."""
        features = batch["features"].to(self.device)
        train_indices = batch["train_indices"].to(self.device)
        test_indices = batch["test_indices"].to(self.device)
        train_labels = batch["train_labels"].to(self.device)
        test_labels = batch["test_indices"].to(self.device)

        # Forward pass
        logits = self.model(features, train_indices, test_indices, train_labels)

        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, test_labels)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Check gradients at each parameter
        print("\n" + "=" * 60)
        print("GRADIENT FLOW DIAGNOSTIC")
        print("=" * 60)

        print(f"\n1. LOSS INFO")
        print(f"   Loss value: {loss.item():.6f}")
        print(f"   Loss grad exists: {loss.grad is not None}")

        print(f"\n2. PARAMETER GRADIENTS")
        zero_grad_params = []
        nonzero_grad_params = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.grad_norms[name] = grad_norm
                if grad_norm < 1e-10:
                    zero_grad_params.append((name, grad_norm))
                else:
                    nonzero_grad_params.append((name, grad_norm))
            else:
                zero_grad_params.append((name, 0.0))

        print(f"\n   Parameters with ZERO gradients ({len(zero_grad_params)}):")
        for name, norm in zero_grad_params:
            print(f"     {name}: {norm:.2e}")

        print(f"\n   Parameters with NON-ZERO gradients ({len(nonzero_grad_params)}):")
        for name, norm in nonzero_grad_params[:10]:  # Show first 10
            print(f"     {name}: {norm:.6f}")
        if len(nonzero_grad_params) > 10:
            print(f"     ... and {len(nonzero_grad_params) - 10} more")

        print(f"\n3. LAYER-WISE GRADIENT ANALYSIS")

        # Check specific layers
        layers_to_check = [
            ("feature_embedding", self.model.pfn.feature_embedding),
            ("position_embedding", self.model.pfn.position_embedding),
            ("cls_token", self.model.pfn.cls_token),
            ("classification_head", self.model.pfn.classification_head),
            ("transformer", self.model.pfn.transformer),
        ]

        for layer_name, layer in layers_to_check:
            if hasattr(layer, 'named_parameters'):
                layer_grad_norms = []
                for n, p in layer.named_parameters():
                    if p.grad is not None:
                        layer_grad_norms.append(p.grad.norm().item())
                if layer_grad_norms:
                    total_norm = sum(g**2 for g in layer_grad_norms)**0.5
                    print(f"   {layer_name}: grad_norm = {total_norm:.6f}")
                else:
                    print(f"   {layer_name}: NO GRADIENTS")

        print(f"\n4. LOGIT GRADIENT CHECK")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits requires_grad: {logits.requires_grad}")
        print(f"   Logits grad_fn: {logits.grad_fn}")

        print(f"\n5. INPUT GRADIENT CHECK")
        print(f"   Features shape: {features.shape}")
        print(f"   Features requires_grad: {features.requires_grad}")

        # Check if loss depends on logits
        print(f"\n6. GRADIENT COMPUTATION CHECK")
        print(f"   Loss depends on: {[n for n, _ in self.model.named_parameters() if _.requires_grad]}")

        print("\n" + "=" * 60)
        print("DIAGNOSIS COMPLETE")
        print("=" * 60)

        return self.grad_norms


def run_gradient_diagnostic():
    print("=" * 60)
    print("Gradient Flow Diagnostic")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model and data
    model = BayesPFNv1(n_features=32, d_model=128, n_heads=4, n_layers=3, n_classes=2)
    model.to(device)
    model.train()

    dataset = DiskICLDataset(data_dir="./data/synthetic")
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_disk_batch)

    batch = next(iter(loader))

    checker = GradientChecker(model)

    # Run diagnostic
    grad_norms = checker.check_gradients(batch)

    return grad_norms


if __name__ == "__main__":
    grad_norms = run_gradient_diagnostic()
