#!/usr/bin/env bash
# =============================================================================
# flair2_unlock.sh — full flair2 GPU unlock sequence (Agent G, Wave 1 Day 0-1).
#
# Source: docs/MASTER_PROTOCOL_AUG10.md
#   Part 2  — "flair2 unlock (Agent G, Day 0-1, run on ethernet or overnight)"
#             (lines ~68-86): the exact 14 nvidia-cu12 wheel pins + rsync + venv
#             + offline pip install + CUDA gate ("must print True 2").
#   Part 3  — "AGENT G — flair2 unlock" (lines ~187-189): gate is
#             torch.cuda.is_available() -> True, device_count -> 2. Do not start
#             Wave 2's server work until this gate passes.
#
# This script runs the ENTIRE unlock sequence end to end:
#   STEP 1  download the 14 nvidia-cu12 + triton wheels into wheelhouse/ on Mac
#           (idempotent: pip download skips already-present wheels)
#   STEP 2  rsync wheelhouse/ -> flair2:/data/srujan.sai/wheelhouse/
#   STEP 3  ssh flair2: create venv_gpu, pip install --no-index from the wheelhouse
#           (torch torchvision numpy scipy scikit-learn pandas)
#   STEP 4  GATE CHECK: python -c "import torch;print(torch.cuda.is_available(),
#           torch.cuda.device_count())"  MUST print "True 2"
#
# Idempotent: every step is safe to re-run. Uses set -euo pipefail.
#
# USAGE:
#   bash scripts/flair2_unlock.sh                 # uses ssh Host "flair2"
#   FLAIR2_REMOTE=srujan.sai@flair2.iitgn.ac.in bash scripts/flair2_unlock.sh
#
# NOTE: STEP 1 touches the network (PyPI). STEP 2/3 touch flair2 over SSH.
#       If you are offline, STEP 1 will fail fast — pre-populate wheelhouse/
#       manually before re-running. The script never deletes wheels it already
#       has, so a partial download is resumable.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- Remote target -----------------------------------------------------------
# Default uses the ssh-config Host alias "flair2" (see
# scripts/flair2_ssh_config_snippet.txt). Override with FLAIR2_REMOTE for an
# explicit user@host.
REMOTE="${FLAIR2_REMOTE:-flair2}"
REMOTE_WHEELHOUSE="${FLAIR2_REMOTE_WHEELHOUSE:-/data/srujan.sai/wheelhouse}"
REMOTE_PROJECT="${FLAIR2_REMOTE_PROJECT:-/data/srujan.sai/DRO-FairML-run}"
VENV_DIR="venv_gpu"

# --- The 14 nvidia-cu12 wheels + triton (EXACT pins from MASTER_PROTOCOL Part 2)
NVIDIA_WHEELS=(
  "nvidia-cuda-nvrtc-cu12==12.4.127"
  "nvidia-cuda-runtime-cu12==12.4.127"
  "nvidia-cuda-cupti-cu12==12.4.127"
  "nvidia-cudnn-cu12==9.1.0.70"
  "nvidia-cublas-cu12==12.4.5.8"
  "nvidia-cufft-cu12==11.2.1.3"
  "nvidia-curand-cu12==10.3.5.147"
  "nvidia-cusolver-cu12==11.6.1.9"
  "nvidia-cusparse-cu12==12.3.1.170"
  "nvidia-cusparselt-cu12==0.6.2"
  "nvidia-nccl-cu12==2.21.5"
  "nvidia-nvtx-cu12==12.4.127"
  "nvidia-nvjitlink-cu12==12.4.127"
  "triton==3.2.0"
)

echo "================================================================"
echo " flair2 GPU unlock  (Agent G, MASTER_PROTOCOL_AUG10.md Part 2)"
echo "   remote         : ${REMOTE}"
echo "   wheelhouse     : ${REMOTE_WHEELHOUSE}"
echo "   project        : ${REMOTE_PROJECT}"
echo "   venv           : ${VENV_DIR}"
echo "   nvidia wheels  : ${#NVIDIA_WHEELS[@]} (14 per Part 2)"
echo "================================================================"

# =============================================================================
# STEP 1 — Finish downloading the 14 nvidia-cu12 wheels to wheelhouse/ on Mac.
#   Idempotent: pip download skips wheels already present in wheelhouse/.
# =============================================================================
echo ""
echo "=== STEP 1: download 14 nvidia-cu12 + triton wheels into wheelhouse/ ==="
mkdir -p wheelhouse
echo "--> target: ${ROOT}/wheelhouse ($(ls -1 wheelhouse/*.whl 2>/dev/null | wc -l | tr -d ' ') wheels already present)"

# pip download with PEP 425 platform/abi tags so macOS can fetch linux wheels
# for the flair2 node (manylinux2014_x86_64, cp310). --no-deps because we are
# pinning the exact nvidia set; the torch wheel (already in wheelhouse if the
# 830 MB pre-download exists) pulls these as deps on the server instead.
python3 -m pip download \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  --abi cp310 \
  --only-binary=:all: \
  --no-deps \
  -d wheelhouse \
  "${NVIDIA_WHEELS[@]}"

echo "--> wheelhouse now contains $(ls -1 wheelhouse/*.whl 2>/dev/null | wc -l | tr -d ' ') wheels"

# Sanity: confirm each of the 14 expected wheels is present (by name stem).
echo "--> verifying all 14 nvidia/triton wheels are present:"
MISSING=0
for pkg in "${NVIDIA_WHEELS[@]}"; do
  name="${pkg%%==*}"                       # e.g. nvidia-cuda-nvrtc-cu12
  if ls wheelhouse/"${name}"-*.whl >/dev/null 2>&1; then
    echo "    [OK]   ${name}: $(ls wheelhouse/"${name}"-*.whl 2>/dev/null | head -1 | xargs -n1 basename)"
  else
    echo "    [MISS] ${name}: NOT FOUND in wheelhouse/"
    MISSING=$((MISSING + 1))
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "FATAL: ${MISSING} of 14 wheels still missing after download. Aborting before rsync."
  exit 1
fi
echo "STEP 1 OK — all 14 nvidia-cu12 + triton wheels present in wheelhouse/."

# =============================================================================
# STEP 2 — rsync wheelhouse/ to flair2:/data/srujan.sai/wheelhouse/
# =============================================================================
echo ""
echo "=== STEP 2: rsync wheelhouse/ -> ${REMOTE}:${REMOTE_WHEELHOUSE}/ ==="
# Ensure the remote wheelhouse dir exists, then sync (idempotent; --update skips
# files that are already newer on the destination).
ssh "$REMOTE" "mkdir -p '${REMOTE_WHEELHOUSE}'"
rsync -az --update --progress \
  wheelhouse/ "${REMOTE}:${REMOTE_WHEELHOUSE}/"
echo "STEP 2 OK — wheelhouse synced to ${REMOTE}:${REMOTE_WHEELHOUSE}/."

# =============================================================================
# STEP 3 — ssh to flair2: create venv_gpu + pip install --no-index from wheelhouse
#   (torch torchvision numpy scipy scikit-learn pandas)
# =============================================================================
echo ""
echo "=== STEP 3: create ${VENV_DIR} + offline pip install from wheelhouse ==="
ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
cd '${REMOTE_PROJECT}'
if [ ! -x ${VENV_DIR}/bin/python3 ]; then
  echo "--> creating venv at ${VENV_DIR}/"
  python3 -m venv ${VENV_DIR}
fi
echo "--> upgrading pip (offline-safe: falls back to bundled)"
${VENV_DIR}/bin/pip install --upgrade pip || true
echo "--> installing torch stack from wheelhouse (NO INDEX = offline)"
${VENV_DIR}/bin/pip install --no-index \
  --find-links '${REMOTE_WHEELHOUSE}' \
  torch torchvision numpy scipy scikit-learn pandas
echo "STEP 3 OK — offline install complete."
EOF

# =============================================================================
# STEP 4 — GATE CHECK: torch.cuda.is_available() + device_count() must be True 2
# =============================================================================
echo ""
echo "=== STEP 4: GATE CHECK — must print 'True 2' ==="
GATE_OUT="$(ssh "$REMOTE" "${VENV_DIR_ABS:-${REMOTE_PROJECT}/${VENV_DIR}}/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'")"
echo "gate output: ${GATE_OUT}"

# Normalize whitespace and compare. Expected exactly: "True 2"
NORMALIZED="$(echo "$GATE_OUT" | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')"
if [ "$NORMALIZED" = "True 2" ]; then
  echo "GATE PASSED — torch.cuda.is_available()=True, device_count=2."
  echo "flair2 is UNLOCKED. Wave 2 (Agent U) may proceed."
  exit 0
else
  echo "GATE FAILED — expected 'True 2', got '${NORMALIZED}'."
  echo "Do NOT start Wave 2 server work. Diagnose CUDA/driver/wheel mismatch."
  exit 2
fi