#!/bin/bash
#===============================================================================
# BayesPFN Environment Setup Script
# 
# Purpose : Automates Phase 1-4 for Vast.AI instance setup
# Author  : Bayu Satya Adhitama
# Date    : April 2026
#
# Usage:
#   bash setup/setup.sh [--skip-torch] [--skip-r2] [--verbose]
#
#===============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TABPFN_REPO="https://github.com/automl/TabPFN.git"
BAYESPFN_REPO=""  # Set this to your BayesPFN repo URL if different

# Parse arguments
SKIP_TORCH=false
SKIP_R2=false
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --skip-torch) SKIP_TORCH=true ;;
        --skip-r2) SKIP_R2=true ;;
        --verbose) VERBOSE=true ;;
    esac
done

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

run_cmd() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[CMD]${NC} $1"
    fi
    eval $1
}

#===============================================================================
# PHASE 1: System Environment Check
#===============================================================================
phase1_system() {
    log_info "========================================"
    log_info "Phase 1: System Environment Check"
    log_info "========================================"
    
    # Check Python version
    log_info "Checking Python version..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        log_success "Python $PYTHON_VERSION - OK (>= 3.9 required)"
    else
        log_error "Python $PYTHON_VERSION - Need Python 3.9 or higher"
        exit 1
    fi
    
    # Check NVIDIA drivers
    log_info "Checking NVIDIA drivers..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
        log_success "NVIDIA drivers found"
    else
        log_warn "nvidia-smi not found - GPU monitoring will not be available"
    fi
    
    # Check CUDA
    log_info "Checking CUDA..."
    if [ -d "/usr/local/cuda" ] || [ -d "/usr/lib/cuda" ]; then
        log_success "CUDA installation found"
    else
        log_warn "CUDA not found in standard location"
    fi
    
    # Check git
    log_info "Checking git..."
    if command -v git &> /dev/null; then
        log_success "git $(git --version | awk '{print $3}')"
    else
        log_error "git not found"
        exit 1
    fi
    
    # Check disk space
    log_info "Checking disk space..."
    DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
    log_success "Available disk space: $DISK_AVAIL"
}

#===============================================================================
# PHASE 2: PyTorch Installation
#===============================================================================
phase2_pytorch() {
    log_info "========================================"
    log_info "Phase 2: PyTorch Installation"
    log_info "========================================"
    
    if [ "$SKIP_TORCH" = true ]; then
        log_warn "Skipping PyTorch installation (--skip-torch)"
        return
    fi
    
    # Check if PyTorch already installed with CUDA
    log_info "Checking existing PyTorch installation..."
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null | grep -q "True"; then
            log_success "PyTorch with CUDA already installed"
            return
        fi
    fi
    
    # Detect CUDA version
    CUDA_VERSION=""
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
    fi
    
    if [ -z "$CUDA_VERSION" ]; then
        log_warn "Could not detect CUDA version, defaulting to cu121"
        CUDA_VERSION="12.1"
    fi
    
    log_info "Installing PyTorch with CUDA $CUDA_VERSION..."
    
    # Install PyTorch
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION%%.*}xx \
        --quiet
    
    # Verify installation
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
    
    log_success "PyTorch installation complete"
}

#===============================================================================
# PHASE 3: Repository Setup
#===============================================================================
phase3_repos() {
    log_info "========================================"
    log_info "Phase 3: Repository Setup"
    log_info "========================================"
    
    # Clone TabPFN if not exists
    if [ ! -d "TabPFN" ]; then
        log_info "Cloning TabPFN v2 repository..."
        git clone $TABPFN_REPO TabPFN
        log_success "TabPFN cloned"
    else
        log_info "TabPFN already exists, updating..."
        cd TabPFN && git pull && cd ..
        log_success "TabPFN updated"
    fi
    
    # Install TabPFN dependencies
    log_info "Installing TabPFN dependencies..."
    cd TabPFN
    pip install -e . --quiet
    cd ..
    
    # Verify TabPFN import
    python3 -c "from tabpfn import TabPFNClassifier; print('TabPFN import: OK')"
    log_success "TabPFN installed"
    
    # Setup BayesPFN directory if needed
    if [ ! -d "bayespfn" ]; then
        log_info "Creating bayespfn project directory..."
        mkdir -p bayespfn
        cd bayespfn
        git init
        git remote add origin "$BAYESPFN_REPO" 2>/dev/null || true
        cd ..
        log_info "bayespfn directory created"
    fi
    
    log_success "Repository setup complete"
}

#===============================================================================
# PHASE 4: Cloud Storage (R2) Setup
#===============================================================================
phase4_storage() {
    log_info "========================================"
    log_info "Phase 4: Cloud Storage (R2) Setup"
    log_info "========================================"
    
    if [ "$SKIP_R2" = true ]; then
        log_warn "Skipping R2 setup (--skip-r2)"
        return
    fi
    
    # Install boto3
    log_info "Installing boto3..."
    pip install boto3 --quiet
    python3 -c "import boto3; print('boto3 version:', boto3.__version__)"
    log_success "boto3 installed"
    
    # Create R2 configuration helper
    log_info "========================================"
    log_info "R2 Configuration Required"
    log_info "========================================"
    
    R2_SETUP_SCRIPT="bayespfn/r2_config.py"
    
    cat > $R2_SETUP_SCRIPT << 'R2EOF'
#!/usr/bin/env python3
"""
R2 Configuration Helper for BayesPFN

Usage:
    python r2_config.py --configure     # Interactive setup
    python r2_config.py --test         # Test connection
    python r2_config.py --list         # List buckets
"""

import os
import boto3
from botocore.exceptions import ClientError
import json

def load_config():
    """Load R2 config from environment or .env file"""
    config = {
        'account_id': os.getenv('R2_ACCOUNT_ID', ''),
        'access_key_id': os.getenv('R2_ACCESS_KEY_ID', ''),
        'secret_access_key': os.getenv('R2_SECRET_ACCESS_KEY', ''),
        'bucket_name': os.getenv('R2_BUCKET_NAME', 'bayespfn-checkpoints'),
    }
    return config

def save_config(config):
    """Save config to .env file"""
    with open('.env', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}={value}\n')
    print("Configuration saved to .env")

def configure():
    """Interactive R2 configuration"""
    print("=" * 50)
    print("R2 Configuration Setup")
    print("=" * 50)
    
    config = load_config()
    
    print("\nEnter your Cloudflare R2 credentials:")
    print("(Press Enter to keep current value)\n")
    
    # Account ID
    val = input(f"R2 Account ID [{config['account_id']}]: ").strip()
    if val: config['account_id'] = val
    
    # Access Key ID
    val = input(f"R2 Access Key ID [{config['access_key_id']}]: ").strip()
    if val: config['access_key_id'] = val
    
    # Secret Access Key
    val = input(f"R2 Secret Access Key [{config['secret_access_key'][:10] if config['secret_access_key'] else ''}...]: ").strip()
    if val: config['secret_access_key'] = val
    
    # Bucket Name
    val = input(f"R2 Bucket Name [{config['bucket_name']}]: ").strip()
    if val: config['bucket_name'] = val
    
    save_config(config)
    print("\nConfiguration saved!")
    
    # Verify
    if test_connection(config):
        print("Connection verified successfully!")
    else:
        print("Warning: Connection test failed. Check your credentials.")

def test_connection(config=None):
    """Test R2 connection"""
    if config is None:
        config = load_config()
    
    try:
        session = boto3.Session(
            aws_access_key_id=config['access_key_id'],
            aws_secret_access_key=config['secret_access_key'],
            region_name='auto'
        )
        
        s3 = session.client(
            's3',
            endpoint_url=f"https://{config['account_id']}.r2.cloudflarestorage.com"
        )
        
        s3.list_buckets()
        print("R2 connection: OK")
        return True
        
    except ClientError as e:
        print(f"R2 connection failed: {e}")
        return False

def create_bucket_structure():
    """Create recommended R2 bucket structure"""
    config = load_config()
    
    try:
        session = boto3.Session(
            aws_access_key_id=config['access_key_id'],
            aws_secret_access_key=config['secret_access_key'],
            region_name='auto'
        )
        
        s3 = session.client(
            's3',
            endpoint_url=f"https://{config['account_id']}.r2.cloudflarestorage.com"
        )
        
        # Create folder structure
        prefixes = [
            'checkpoints/bayespfn_v1/',
            'checkpoints/bayespfn_v2/',
            'checkpoints/bayespfn_v3/',
            'logs/',
            'data/',
        ]
        
        for prefix in prefixes:
            s3.put_object(Bucket=config['bucket_name'], Key=prefix)
            print(f"Created: {prefix}")
        
        print("Bucket structure created!")
        
    except ClientError as e:
        print(f"Error creating bucket structure: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--configure':
            configure()
        elif sys.argv[1] == '--test':
            test_connection()
        elif sys.argv[1] == '--list':
            config = load_config()
            session = boto3.Session(
                aws_access_key_id=config['access_key_id'],
                aws_secret_access_key=config['secret_access_key'],
                region_name='auto'
            )
            s3 = session.client(
                's3',
                endpoint_url=f"https://{config['account_id']}.r2.cloudflarestorage.com"
            )
            print("Buckets:", [b['Name'] for b in s3.list_buckets()['Buckets']])
        elif sys.argv[1] == '--init-structure':
            create_bucket_structure()
        else:
            print("Unknown option:", sys.argv[1])
    else:
        print(__doc__)
R2EOF
    
    chmod +x $R2_SETUP_SCRIPT
    log_success "R2 config helper created: $R2_SETUP_SCRIPT"
    log_info ""
    log_info "To configure R2, run:"
    log_info "  python3 $R2_SETUP_SCRIPT --configure"
    log_info ""
    log_info "To test connection:"
    log_info "  python3 $R2_SETUP_SCRIPT --test"
}

#===============================================================================
# PHASE 5: Development Tools
#===============================================================================
phase5_devtools() {
    log_info "========================================"
    log_info "Phase 5: Development Tools"
    log_info "========================================"
    
    # Install JupyterLab
    log_info "Installing JupyterLab..."
    pip install jupyterlab --quiet
    log_success "JupyterLab installed"
    
    # Install Weights & Biases
    log_info "Installing Weights & Biases..."
    pip install wandb --quiet
    log_success "WandB installed (optional but recommended)"
    
    # Install common useful tools
    log_info "Installing additional tools..."
    pip install \
        ipython \
        htop \
        tmux \
        tree \
        --quiet
    log_success "Additional tools installed"
    
    # Install tabpfn extensions (optional)
    log_info "Optional: Installing tabpfn-extensions..."
    pip install git+https://github.com/priorlabs/tabpfn-extensions.git --quiet || true
}

#===============================================================================
# PHASE 6: Verification Tests
#===============================================================================
phase6_verify() {
    log_info "========================================"
    log_info "Phase 6: Verification Tests"
    log_info "========================================"
    
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Test 1: TabPFN import
    log_info "Test 1: TabPFN import..."
    if python3 -c "from tabpfn import TabPFNClassifier" 2>/dev/null; then
        log_success "TabPFN: PASS"
        ((TESTS_PASSED++))
    else
        log_error "TabPFN: FAIL"
        ((TESTS_FAILED++))
    fi
    
    # Test 2: PyTorch CUDA
    log_info "Test 2: PyTorch CUDA..."
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')")
        log_success "CUDA: PASS (Device: $DEVICE, VRAM: ${VRAM}GB)"
        ((TESTS_PASSED++))
    else
        log_warn "CUDA: SKIP (no GPU or CPU-only mode)"
    fi
    
    # Test 3: boto3/R2
    log_info "Test 3: R2 connection..."
    if python3 -c "import boto3" 2>/dev/null; then
        log_success "boto3: PASS"
        ((TESTS_PASSED++))
    else
        log_error "boto3: FAIL"
        ((TESTS_FAILED++))
    fi
    
    # Test 4: wandb
    log_info "Test 4: Weights & Biases..."
    if python3 -c "import wandb" 2>/dev/null; then
        log_success "wandb: PASS"
        ((TESTS_PASSED++))
    else
        log_warn "wandb: SKIP (optional)"
    fi
    
    # Test 5: Git
    log_info "Test 5: Git..."
    if git remote -v 2>/dev/null | grep -q "bayespfn"; then
        REPO_URL=$(git remote -v | grep bayespfn | head -1 | awk '{print $2}')
        log_success "Git: PASS (remote: $REPO_URL)"
        ((TESTS_PASSED++))
    else
        log_warn "Git: WARN (no remote configured)"
    fi
    
    # Summary
    log_info "========================================"
    log_info "Verification Summary"
    log_info "========================================"
    log_info "Tests Passed: $TESTS_PASSED"
    log_info "Tests Failed: $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All critical tests passed!"
        return 0
    else
        log_error "Some tests failed. Check logs above."
        return 1
    fi
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================
main() {
    log_info "========================================"
    log_info "BayesPFN Setup Script"
    log_info "========================================"
    log_info "Starting setup..."
    log_info ""
    
    # Run all phases
    phase1_system
    phase2_pytorch
    phase3_repos
    phase4_storage
    phase5_devtools
    phase6_verify
    
    log_info ""
    log_info "========================================"
    log_info "Setup Complete!"
    log_info "========================================"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Configure R2: python3 bayespfn/r2_config.py --configure"
    log_info "  2. Initialize R2 structure: python3 bayespfn/r2_config.py --init-structure"
    log_info "  3. Push to your repo: cd bayespfn && git add . && git commit -m 'Initial setup'"
    log_info ""
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
