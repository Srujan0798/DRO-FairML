#!/usr/bin/env bash
# Install CUDA torch offline on flair2 after wheelhouse is synced.
# Usage: bash scripts/flair2_install_gpu.sh
set -euo pipefail

REMOTE="${FLAIR2_REMOTE:-flair2}"
REMOTE_WH="${FLAIR2_REMOTE_WHEELHOUSE:-/data/srujan.sai/wheelhouse}"
REMOTE_PROJ="${FLAIR2_REMOTE_PROJECT:-/data/srujan.sai/DRO-FairML-run}"
VENV=venv_gpu

echo "=== sync project (code + data features if present) ==="
ssh "$REMOTE" "mkdir -p '$REMOTE_PROJ' '$REMOTE_WH'"
rsync -az --delete \
  --exclude '.git' --exclude 'wheelhouse' --exclude '__pycache__' \
  --exclude '.mypy_cache' --exclude 'logs' --exclude '.pytest_cache' \
  --exclude 'paper/main.pdf' --exclude 'report/report.pdf' \
  ./ "${REMOTE}:${REMOTE_PROJ}/"

echo "=== sync wheelhouse ==="
rsync -az --progress wheelhouse/ "${REMOTE}:${REMOTE_WH}/"

echo "=== create venv + offline install ==="
ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
cd '${REMOTE_PROJ}'
python3 -m venv ${VENV}
${VENV}/bin/pip install -U pip setuptools wheel
# Install torch first from wheelhouse
${VENV}/bin/pip install --no-index --find-links '${REMOTE_WH}' \
  torch torchvision 2>&1 | tail -30
# Then scientific stack
${VENV}/bin/pip install --no-index --find-links '${REMOTE_WH}' \
  numpy scipy scikit-learn pandas pillow 2>&1 | tail -20 || true
# If pure-python packages missing from wheelhouse, try online once as fallback
${VENV}/bin/pip install numpy scipy scikit-learn pandas pillow 2>&1 | tail -15 || true

echo "=== GATE ==="
${VENV}/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
nvidia-smi -L
EOF

echo "DONE"
