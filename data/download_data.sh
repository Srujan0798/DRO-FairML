#!/usr/bin/env bash
#
# download_data.sh — fetch all required datasets for a fresh clone.
#
# The repository ships NO data (see .gitignore: data/raw/ is ignored). Without this
# step nothing in experiments/ or tests/ that touches real data can run. Tabular
# datasets (Adult, Credit, LSAC) are downloaded automatically; UTKFace images must
# be fetched manually (large, and GPU access was never granted for this project).
#
# Usage:
#   bash data/download_data.sh            # tabular only (Adult/Credit/LSAC)
#   bash data/download_data.sh --utkface  # also prints UTKFace manual-fetch steps
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ">> Downloading tabular datasets into data/raw/ ..."
python3 - <<'PY'
from src.data.datasets import load_adult, load_credit, load_lsac
for loader in (load_adult, load_credit, load_lsac):
    try:
        loader(data_dir="data/raw")
        print(f"  OK: {loader.__name__}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {loader.__name__}: {e}")
        raise SystemExit(1)
PY

if [[ "${1:-}" == "--utkface" ]]; then
    echo
    echo ">> UTKFace (manual download required):"
    echo "   1. Download the aligned+cropped UTKFace archive from:"
    echo "        https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUxOckljUUU"
    echo "   2. Unpack into: data/raw/utkface/  (files named *.jpg.chip.jpg)"
    echo "   3. Optionally pre-extract features: python3 scripts/extract_utkface_features.py"
    echo "   GPU access for UTKFace was never granted; only a CPU smoke path exists."
else
    echo
    echo ">> Tabular data ready. Re-run with --utkface for UTKFace manual-fetch steps."
fi

echo ">> Done."
