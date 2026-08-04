#!/bin/bash
# Setup script for UTKFace experiments on GPU server
# Run this after getting GPU server access

set -e

echo "=== UTKFace GPU Server Setup ==="

# Config
SERVER_USER="srujan.sai"
SERVER_HOST="flair2.iitgn.ac.in"  # Update with actual hostname when confirmed
REMOTE_DATA_DIR="/data/${SERVER_USER}/UTKFace"
REMOTE_CACHE="/data/${SERVER_USER}/utkface_features.npz"
PROJECT_DIR="/home/${SERVER_USER}/DRO-FairML"

echo "Server: ${SERVER_USER}@${SERVER_HOST}"
echo "Data dir: ${REMOTE_DATA_DIR}"

# 1. Create directories
echo "[1/5] Creating remote directories..."
ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DATA_DIR} ${PROJECT_DIR}"

# 2. Copy project code
echo "[2/5] Copying project code..."
rsync -avz --exclude='venv' --exclude='data/raw' --exclude='figures' --exclude='results' \
    /Users/srujansai/Desktop/DRO-FairML/ \
    ${SERVER_USER}@${SERVER_HOST}:${PROJECT_DIR}/

# 3. Download UTKFace (if not present)
echo "[3/5] Checking UTKFace data..."
ssh ${SERVER_USER}@${SERVER_HOST} << EOF
    if [ ! -f "${REMOTE_DATA_DIR}/1.jpg.chip.jpg" ]; then
        echo "UTKFace not found. Downloading..."
        # Option 1: wget from mirror
        cd /tmp
        wget -q https://github.com/moo-simple-unet/releases/download/v1.0/UTKFace.tar.gz -O UTKFace.tar.gz 2>/dev/null || echo "wget failed, use manual upload"
        # Option 2: kaggle (requires API key)
        # kaggle datasets download -d jangedoo/utkface-new
        echo "If auto-download failed, manually upload UTKFace images to ${REMOTE_DATA_DIR}"
    else
        echo "UTKFace already present"
    fi
EOF

# 4. Extract features
echo "[4/5] Extracting ResNet18 features..."
ssh ${SERVER_USER}@${SERVER_HOST} << EOF
    cd ${PROJECT_DIR}
    python3 scripts/extract_utkface_features.py \
        --data-dir ${REMOTE_DATA_DIR} \
        --output ${REMOTE_CACHE} \
        --batch-size 128
EOF

# 5. Run smoke test
echo "[5/5] Running UTKFace smoke test..."
ssh ${SERVER_USER}@${SERVER_HOST} << EOF
    cd ${PROJECT_DIR}
    python3 experiments/run_utkface.py --smoke
EOF

echo "=== Setup complete ==="
echo "Next: run full experiments with:"
echo "  ssh ${SERVER_USER}@${SERVER_HOST} 'cd ${PROJECT_DIR} && python3 experiments/run_utkface.py --alphas 0.0 0.1 0.2 0.3 --n_seeds 5'"
