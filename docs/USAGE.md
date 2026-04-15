# BayesPFN-v1 User Guide

## Overview

BayesPFN is a foundation model for tabular classification that extends TabPFN with **Innovation 1: Imbalance-Stratified Pretraining**. Instead of TabPFN's uniform prior that produces uncontrolled imbalance distributions, BayesPFN uses a stratified zone sampling approach to ensure the model learns to handle extreme class imbalance.

**Current Branch:** `feature/bayespfn-v1-imbalance-prior`

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bayespfn.git
cd bayespfn

# Switch to BayesPFN-v1 branch
git checkout feature/bayespfn-v1-imbalance-prior

# Install dependencies
pip install torch numpy scikit-learn tqdm pyyaml wandb
```

### 2. Generate Synthetic Data

```bash
# Verify stratified zone sampling (Zone A/B/C distribution)
python scripts/generate_data.py --n-datasets 500 --verify

# Generate 500 datasets (sanity check)
python scripts/generate_data.py --n-datasets 500 --output-dir ./data/synthetic

# Generate 25,000 datasets (full pretraining)
python scripts/generate_data.py --n-datasets 25000 --output-dir ./data/synthetic
```

### 3. Train Model

```bash
# Train on 500 datasets (sanity check, ~30 minutes)
python scripts/train.py --n-datasets 500

# Train on 25,000 datasets (full run, ~3-4 hours on RTX 5090)
python scripts/train.py --n-datasets 25000
```

### 4. Evaluate

```bash
# Evaluate on benchmark datasets
python scripts/evaluate.py --checkpoint ./checkpoints/bayespfn_v1_final.ckpt --dataset all
```

---

## Project Structure

```
bayespfn/
├── src/
│   ├── __init__.py          # Package exports
│   ├── imbalance.py         # StratifiedZoneSampler
│   ├── generator.py         # SyntheticDataGenerator + ICLDataset
│   ├── model.py             # BayesPFNv1 + PFNTransformer
│   ├── trainer.py          # Trainer class
│   └── evaluation.py       # Evaluator class
├── configs/
│   └── bayespfn_v1.yaml    # Hyperparameters
├── scripts/
│   ├── generate_data.py    # Data generation
│   ├── train.py           # Training entry point
│   └── evaluate.py        # Evaluation
├── checkpoints/           # Saved model checkpoints
└── logs/                  # Training logs
```

---

## Core Concepts

### Stratified Zone Sampling (Innovation 1)

Instead of using a single Beta distribution (which cannot satisfy both constraints), we partition the imbalance ratio space into three zones:

| Zone | Imbalance Ratio | Target % | Sampling Method |
|------|----------------|----------|-----------------|
| Zone A | r ∈ [1, 5) | 60% | Uniform |
| Zone B | r ∈ [5, 10) | 10% | Uniform |
| Zone C | r ∈ [10, 100] | 30% | Power-law |

**Why this matters:** TabPFN's uniform prior over SCMs implicitly produces mostly moderate imbalance. BayesPFN ensures 30% of pretraining datasets have severe imbalance (r > 10:1), forcing the model to internalize minority-class patterns.

### Synthetic Data Generation

The synthetic generator creates datasets that approximate TabPFN's prior:

1. **Features:** Gaussian mixtures + polynomial interactions (simulates causal structure)
2. **Labels:** GradientBoostingClassifier as ground truth (simulates Bayesian posterior)
3. **Imbalance:** Subsample majority class to achieve target minority proportion π

### In-Context Learning Format

BayesPFN uses the same ICL format as TabPFN:

```
[CLS] | [x_1^train features] | [x_2^train features] | ... | [x_n_train^train features] | [y_1] | [y_2] | ... | [y_n_train] | [x_1^test features] | ...
         ←─────── D_train (80%) ────────→ |←── D_cal (20%) ──→ | ←── D_test ──→
```

---

## Configuration

Edit `configs/bayespfn_v1.yaml` to adjust hyperparameters:

```yaml
bayespfn_v1:
  model:
    n_features: 32        # Embedding dimension
    d_model: 128         # Model dimension
    n_heads: 4           # Attention heads
    n_layers: 3          # Transformer layers
    n_classes: 2         # Binary classification

  training:
    learning_rate: 1.0e-4
    batch_size: 8
    num_epochs: 10

  data:
    n_features: 32
    n_samples_range: [500, 2000]

    imbalance:           # Stratified zones
      zone_a_ratio: [1.0, 5.0]
      zone_b_ratio: [5.0, 10.0]
      zone_c_ratio: [10.0, 100.0]
      zone_proportions: [0.60, 0.10, 0.30]
```

---

## API Usage

### Python API

```python
from src import (
    StratifiedZoneSampler,
    SyntheticDataGenerator,
    BayesPFNv1,
    Trainer,
    Evaluator,
)

# Create components
sampler = StratifiedZoneSampler()
generator = SyntheticDataGenerator()
model = BayesPFNv1(n_features=32, d_model=128, n_layers=3)

# Generate a dataset
pi = sampler.sample_minority_proportion()
X, y = generator.generate_dataset(pi=pi)

# Evaluate
from src.evaluation import Evaluator
evaluator = Evaluator(model)
metrics = evaluator.evaluate_single(X_train, y_train, X_test, y_test)
```

### Loading a Trained Model

```python
from src.evaluation import load_model_from_checkpoint

model = load_model_from_checkpoint(
    "./checkpoints/bayespfn_v1_final.ckpt",
    n_features=32,
    d_model=128,
    device="cuda"
)
```

---

## Expected Results

### Zone Sampling Verification

When running `--verify` on the data generator:

```
Zone distribution (n=100000):
  Zone A (r < 5:1):   60.0% (target: 60%)
  Zone B (5:1 ≤ r < 10:1): 10.0% (target: 10%)
  Zone C (r ≥ 10:1):  30.0% (target: 30%)
```

### Training Metrics

During training, monitor in Weights & Biases:

| Metric | Expected Range |
|--------|---------------|
| Loss | 0.3 - 0.7 |
| Accuracy | 70% - 90% |

---

## Benchmark Datasets

Evaluation targets:

| Dataset | OpenML ID | Imbalance Ratio |
|---------|-----------|-----------------|
| Creditcard Fraud | 1597 | 577:1 |
| Mammography | 43893 | 42:1 |
| Yeast | 181 | 31:1 |

**Target Metrics:**

- Balanced Accuracy > 0.80
- F1-Macro > 0.75
- Coverage Gap < 0.10

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model size in config:

```yaml
training:
  batch_size: 4  # Reduce from 8

model:
  d_model: 64    # Reduce from 128
  n_layers: 2    # Reduce from 3
```

### Training Loss Not Decreasing

Check:
1. Learning rate (try 1e-3 or 1e-5)
2. Model dimension matches n_features in data
3. GPU is available (`torch.cuda.is_available()`)

### Low GPU Utilization (< 30%)

If GPU utilization is low during training, the data loading pipeline is likely bottlenecked:

1. **Increase num_workers** in DataLoader (in `scripts/train.py`):
   ```python
   loader = torch.utils.data.DataLoader(
       dataset,
       batch_size=train_config["batch_size"],
       collate_fn=collate_icl_batch,
       num_workers=4,        # Increase from 0
       pin_memory=True,      # Enable for faster CPU→GPU transfer
       shuffle=True,
   )
   ```

2. **Monitor improvement**: GPU utilization should jump to 70%+

### Evaluation Fails

Ensure:
1. Checkpoint matches model architecture in config
2. n_features in checkpoint matches your model config

---

## Next Steps (Innovation 2 & 3)

**Innovation 2:** Dual-head transformer with epistemic uncertainty head
- Add second head predicting σ² (variance)
- Train with auxiliary NLL loss

**Innovation 3:** In-context class-conditional conformal calibration
- Split ICL window into D_train (80%) and D_cal (20%)
- Compute per-class nonconformity scores
- Guarantee P(y ∈ C(x) | y=k) ≥ 1-α

---

## Git Workflow

```bash
# View current branch
git branch

# Stage changes
git add src/ configs/ scripts/

# Commit with descriptive message
git commit -m "feat: describe changes"

# Push to remote
git push origin feature/bayespfn-v1-imbalance-prior
```

---

## References

- [TabPFN Paper](https://arxiv.org/abs/2207.06548)
- [TabPFN Nature 2025](https://www.nature.com/articles/s41586-024-08328-0)
- Proposal: `files/proposal.md`
