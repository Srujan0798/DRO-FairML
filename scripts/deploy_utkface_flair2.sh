#!/usr/bin/env bash
# Deploy + launch UTKFace canonical grid on flair2 GPU.
# USAGE:
#   export FLAIR2_USER=your.username   # required if not srujan.sai
#   bash scripts/deploy_utkface_flair2.sh
#
# Prerequisites: ssh works:  ssh ${FLAIR2_USER}@flair2.iitgn.ac.in  (or Host flair2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
USER_NAME="${FLAIR2_USER:-srujan.sai}"
HOST="${FLAIR2_HOST:-10.0.62.234}"
REMOTE_DIR="${FLAIR2_DIR:-/data/${USER_NAME}/DRO-FairML}"
REMOTE="${USER_NAME}@${HOST}"

echo "=== Testing SSH as ${REMOTE} ==="
ssh -o BatchMode=yes -o ConnectTimeout=12 "$REMOTE" 'hostname; whoami; nvidia-smi -L 2>/dev/null | head -5 || echo "no nvidia-smi"'

echo "=== Ensuring remote dirs ==="
ssh "$REMOTE" "mkdir -p ${REMOTE_DIR} /data/${USER_NAME}/UTKFace logs 2>/dev/null; mkdir -p ${REMOTE_DIR}"

echo "=== Rsync project (exclude heavy/local junk) ==="
rsync -az --delete \
  --exclude '.git' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'results/canonical_tau1.json' \
  --exclude 'results/stale_archived' \
  --exclude 'figures' \
  --exclude 'paper/*.pdf' \
  --exclude 'report/*.pdf' \
  --exclude 'data/raw/utkface/' \
  --exclude 'logs' \
  "$ROOT/" "${REMOTE}:${REMOTE_DIR}/"

# Ship real features if present (faster than re-extract on server)
if [ -f data/raw/utkface_features.npz ]; then
  echo "=== Shipping utkface_features.npz ==="
  rsync -az --progress data/raw/utkface_features.npz \
    "${REMOTE}:/data/${USER_NAME}/utkface_features.npz"
  ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}/data/raw && cp -f /data/${USER_NAME}/utkface_features.npz ${REMOTE_DIR}/data/raw/utkface_features.npz 2>/dev/null || true"
fi

echo "=== Remote venv + deps (best effort) ==="
ssh "$REMOTE" bash -s <<EOF
set -e
cd ${REMOTE_DIR}
if [ ! -x venv/bin/python3 ]; then
  python3 -m venv venv || true
fi
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt torch torchvision 2>/dev/null || \
  ./venv/bin/pip install -q numpy scipy scikit-learn torch torchvision tqdm pyyaml 2>/dev/null || true
./venv/bin/python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L || true
EOF

echo "=== Launch UTKFace canonical on GPU (tmux, all 3 attacks) ==="
ssh "$REMOTE" bash -s <<EOF
set -e
cd ${REMOTE_DIR}
mkdir -p logs results
# kill old same-named sessions gently
tmux kill-session -t utk_dp 2>/dev/null || true
tmux kill-session -t utk_if 2>/dev/null || true
tmux kill-session -t utk_comb 2>/dev/null || true
tmux kill-session -t utk_all 2>/dev/null || true

FEAT=/data/${USER_NAME}/utkface_features.npz
[ -f data/raw/utkface_features.npz ] && FEAT=data/raw/utkface_features.npz

# Single aggregate JSON (resume-safe if runner supports it); all 3 attacks on CUDA
# Ensure features path is discoverable (datasets loader looks under data/raw)
export UTKFACE_FEATURES="\$FEAT"
tmux new-session -d -s utk_all "cd ${REMOTE_DIR} && \\
  echo START \$(date) | tee -a logs/utkface_flair2_all.log && \\
  ./venv/bin/python3 experiments/run_utkface_server.py \\
    --attacks dp if combined \\
    --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \\
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \\
    --output results/utkface_canonical.json \\
    2>&1 | tee -a logs/utkface_flair2_all.log; \\
  echo ALL_DONE \$(date) | tee -a logs/utkface_flair2_all.log"

tmux ls
echo "ATTACH: ssh ${REMOTE} -t 'tmux attach -t utk_all'"
echo "LOGS:   ssh ${REMOTE} 'tail -f ${REMOTE_DIR}/logs/utkface_flair2_all.log'"
EOF

echo "=== Deployed. UTKFace running on flair2 GPU. ==="
echo "Note: does NOT touch local results/canonical_tau1.json (IF sweep safe)."
