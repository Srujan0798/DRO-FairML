#!/bin/bash
# Setup script for DRO-FAIR on FLAIR2 server
# Run this ONCE after logging in

echo "=== DRO-FAIR Setup ==="
echo ""

# Install Python packages (try multiple methods)
echo "[1/5] Installing packages..."

if python3 -c "import numpy" 2>/dev/null; then
    echo "numpy already installed"
else
    echo "Installing numpy torch sklearn..."
    pip install numpy torch scikit-learn pandas --quiet 2>&1 || \
    pip install --trusted-host pypi.org numpy torch scikit-learn pandas -i http://pypi.org/simple/ 2>&1 || \
    echo "WARNING: Could not install packages. Check pip manually."
fi

# Check GPU
echo ""
echo "[2/5] GPU Status:"
nvidia-smi 2>&1 | head -15 || echo "nvidia-smi not found"

# Check torch
echo ""
echo "[3/5] PyTorch + CUDA:"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1

# Create data directory
echo ""
echo "[4/5] Setting up directories..."
mkdir -p /data/srujan.sai/DRO-FairML/data/raw
mkdir -p /data/srujan.sai/UTKFace
ls -la /data/srujan.sai/DRO-FairML/

# Copy local data if exists
if [ -d "$(dirname "$0")/../data/raw" ]; then
    echo ""
    echo "[5/5] Copying local data files..."
    cp -n $(dirname "$0")/../data/raw/*.data /data/srujan.sai/DRO-FairML/data/raw/ 2>/dev/null || true
    cp -n $(dirname "$0")/../data/raw/*.xls /data/srujan.sai/DRO-FairML/data/raw/ 2>/dev/null || true
    cp -n $(dirname "$0")/../data/raw/*.csv /data/srujan.sai/DRO-FairML/data/raw/ 2>/dev/null || true
fi

echo ""
echo "=== Setup Complete ==="
echo "Run experiments with:"
echo "  cd /data/srujan.sai/DRO-FairML"
echo "  PYTHONPATH=/data/srujan.sai/DRO-FairML python3 scripts/test_fairness_pgd.py"