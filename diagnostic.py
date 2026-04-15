#!/usr/bin/env python3
"""Comprehensive diagnostic for BayesPFN learning issue."""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from disk_dataset import DiskICLDataset, collate_disk_batch
from torch.utils.data import DataLoader
from model import BayesPFNv1

def run_diagnostics():
    print("=" * 60)
    print("BayesPFN Comprehensive Diagnostic")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load a few datasets
    dataset = DiskICLDataset(data_dir="./data/synthetic")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_disk_batch)
    
    batch = next(iter(loader))
    
    print("\n" + "=" * 40)
    print("1. ICL SEQUENCE CONSTRUCTION")
    print("=" * 40)
    print(f"Features shape: {batch['features'].shape}")
    print(f"Train indices: {batch['train_indices'][:20].cpu().tolist()}")
    print(f"Test indices: {batch['test_indices'][:20].cpu().tolist()}")
    print(f"Train labels: {batch['train_labels'][:10].cpu().tolist()}")
    print(f"Test labels: {batch['test_labels'][:10].cpu().tolist()}")
    print(f"Unique test indices: {batch['test_indices'].unique().numel()}")
    
    # Check for duplicates
    unique_test = batch['test_indices'].unique()
    print(f"Test samples count: {len(batch['test_indices'])}")
    print(f"Unique test indices count: {len(unique_test)}")
    
    print("\n" + "=" * 40)
    print("2. FEATURE VALUES CHECK")
    print("=" * 40)
    features = batch['features']
    test_indices = batch['test_indices']
    
    # Are test features actually distinct?
    test_features = features[test_indices]
    print(f"Test features shape: {test_features.shape}")
    
    # Check if any test features are identical
    unique_features = test_features.unique(dim=0)
    print(f"Unique test feature rows: {unique_features.shape[0]}")
    
    if unique_features.shape[0] < test_features.shape[0]:
        print("WARNING: Some test features are identical!")
    else:
        print("OK: All test features are distinct")
    
    print("\n" + "=" * 40)
    print("3. MODEL INDEXING CHECK")
    print("=" * 40)
    
    model = BayesPFNv1(n_features=32, d_model=128, n_heads=4, n_layers=3, n_classes=2)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        features_tensor = batch['features'].to(device)
        train_indices = batch['train_indices'].to(device)
        test_indices = batch['test_indices'].to(device)
        train_labels = batch['train_labels'].to(device)
        
        # Step through model
        all_indices = torch.cat([train_indices, test_indices])
        print(f"All indices shape: {all_indices.shape}")
        
        x = features_tensor[all_indices]
        print(f"Indexed features shape: {x.shape}")
        
        x_embedded = model.pfn.feature_embedding(x)
        print(f"Embedded features shape: {x_embedded.shape}")
        
        # Check position embeddings
        seq_length = features_tensor.shape[0]
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        print(f"Positions shape: {positions.shape}")
        
        x_pos = x_embedded + model.pfn.position_embedding(positions)
        
        # CLS token
        cls_tokens = model.pfn.cls_token.expand(1, -1, -1)
        print(f"CLS tokens shape: {cls_tokens.shape}")
        
        x_with_cls = torch.cat([cls_tokens, x_pos], dim=1)
        print(f"x with CLS shape: {x_with_cls.shape}")
        
        x_transformed = model.pfn.transformer(x_with_cls)
        print(f"Transformed shape: {x_transformed.shape}")
        
        cls_output = x_transformed[:, 0, :]
        print(f"CLS output shape: {cls_output.shape}")
        
        logits = model.pfn.classification_head(cls_output)
        print(f"Logits shape (before expand): {logits.shape}")
        
        n_test = len(test_indices)
        logits_expanded = logits.expand(n_test, -1)
        print(f"Logits shape (after expand): {logits_expanded.shape}")
        
        print("\n" + "=" * 40)
        print("4. THE CORE ISSUE: CLS TOKEN OUTPUT")
        print("=" * 40)
        
        print(f"CLS output values: {cls_output}")
        print(f"This single CLS output is expanded to ALL {n_test} test samples")
        print("This is why all test samples get the same prediction!")
        
        print("\n" + "=" * 40)
        print("5. VERIFICATION: Are test predictions actually identical?")
        print("=" * 40)
        
        # Run full model
        model_logits = model(features_tensor, train_indices, test_indices, train_labels)
        print(f"Model output logits: {model_logits[:5]}")
        print(f"Are all logits identical? {torch.allclose(model_logits[0], model_logits[1])}")
        
        probs = torch.softmax(model_logits, dim=-1)
        print(f"Probabilities: {probs[:5]}")
        
    print("\n" + "=" * 40)
    print("DIAGNOSIS COMPLETE")
    print("=" * 40)

if __name__ == "__main__":
    run_diagnostics()
