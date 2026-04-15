# BayesPFN Setup

This directory contains automation scripts for setting up the BayesPFN development environment on Vast.AI.

## Contents

| File | Description |
|------|-------------|
| `setup.sh` | Main setup script (Phase 1-6) |
| `r2_utils.py` | R2 checkpoint management utilities |
| `README.md` | This file |

## Quick Start

### Step 1: Clone to Vast.AI Instance

After renting your RTX 5090 instance on Vast.AI:

```bash
# SSH into your instance, then:
git clone https://github.com/YOUR_USERNAME/bayespfn.git
cd bayespfn
```

### Step 2: Run Setup

```bash
# Make scripts executable
chmod +x setup/setup.sh setup/r2_utils.py

# Run full setup (Phase 1-6)
bash setup/setup.sh

# Or with options:
bash setup/setup.sh --verbose    # Show all commands
bash setup/setup.sh --skip-r2    # Skip R2 setup if not needed
```

### Step 3: Configure R2

```bash
# Interactive R2 configuration
python3 setup/r2_utils.py configure

# Or manually with environment variables:
export R2_ACCOUNT_ID="your_account_id"
export R2_ACCESS_KEY_ID="your_access_key"
export R2_SECRET_ACCESS_KEY="your_secret"
export R2_BUCKET_NAME="bayespfn-checkpoints"
```

### Step 4: Initialize R2 Bucket Structure

```bash
python3 setup/r2_utils.py init-structure
```

## Available Scripts

### setup.sh

Main environment setup script.

```bash
bash setup/setup.sh [OPTIONS]

Options:
  --skip-torch   Skip PyTorch installation
  --skip-r2      Skip R2 storage setup
  --verbose      Show all command output
```

### r2_utils.py

R2 checkpoint management utilities.

```bash
# Upload checkpoint
python3 setup/r2_utils.py upload \
    --file ./checkpoints/model.ckpt \
    --key checkpoints/bayespfn_v1/model.ckpt

# Download checkpoint
python3 setup/r2_utils.py download \
    --key checkpoints/bayespfn_v1/model.ckpt \
    --output ./model.ckpt

# List all checkpoints
python3 setup/r2_utils.py list

# List specific version
python3 setup/r2_utils.py list --prefix checkpoints/bayespfn_v1/

# Delete checkpoint
python3 setup/r2_utils.py delete --key checkpoints/bayespfn_v1/model.ckpt
```

## R2 Bucket Structure

```
bayespfn-checkpoints/
├── checkpoints/
│   ├── bayespfn_v1/          # Innovation 1: Imbalance-stratified prior
│   ├── bayespfn_v2/          # Innovation 1+2: + Dual-head
│   └── bayespfn_v3/          # Innovation 1+2+3: Full BayesPFN
├── logs/
│   ├── session1/
│   └── session2/
└── data/
    └── sanity_check/
```

## Verification

After setup, verify everything works:

```bash
# Test TabPFN
python3 -c "from tabpfn import TabPFNClassifier; print('OK')"

# Test GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Test R2
python3 setup/r2_utils.py test
```

## Next Steps

After setup is complete, proceed to the implementation phases:

1. **Session 1**: Baseline verification + Innovation 1 (Imbalance-stratified prior)
2. **Session 2**: Innovations 2+3 + Full evaluation

See the main `docs/` directory for detailed implementation plans.
