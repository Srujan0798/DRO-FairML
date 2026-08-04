#!/usr/bin/env bash
#
# download_data.sh — fetch tabular datasets for a fresh clone (with provenance).
#
# The repository ships NO raw data (see .gitignore: data/raw/ is ignored). Without
# this step, experiments/ and tests/ that touch real data cannot run.
#
# Tabular datasets are public and small (~12 MB total). UTKFace images are large
# and optional; this script does not auto-download them.
#
# Usage:
#   bash data/download_data.sh              # Adult + Credit + LSAC + COMPAS + German
#   bash data/download_data.sh --verify     # re-download if missing, then SHA-256 check
#   bash data/download_data.sh --utkface    # also print UTKFace manual steps
#   bash data/download_data.sh --verify --utkface
#
# Provenance (public sources; loaders in src/data/datasets.py use the same URLs):
#
#   Adult (UCI ML Repository, "Adult" / Census Income)
#     https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
#     https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
#     Local: data/raw/adult.data , data/raw/adult.test
#
#   Credit (UCI "default of credit card clients", Yeh & Lien 2009)
#     https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
#     Local: data/raw/default_of_credit_card_clients.xls
#
#   LSAC bar-passage (public mirror of the law-school dataset)
#     https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv
#     Local: data/raw/lsac.csv
#     Note: not the restricted original LSAC microdata; this is the common public
#     research mirror used by fairness tooling (columns: pass_bar, racetxt, …).
#
#   COMPAS (ProPublica recidivism two-year, public)
#     https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
#     Local: data/raw/compas-scores-two-years.csv
#     Protected attr: race binarized African-American(1) vs Caucasian(0),
#     matching ProPublica's "Machine Bias" and Hardt/Price/Srebro (2016).
#
#   German Credit (UCI statlog/german, public)
#     https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
#     Local: data/raw/german.data
#     Protected attr: sex (1=male, 0=female) per UCI german.doc codebook
#     (A91/A93/A94=male, A92/A95=female). Label: 1=good, 0=bad credit.
#
#   UTKFace (optional, large; NOT fetched here)
#     Aligned+cropped images historically hosted on Google Drive; public Kaggle
#     mirrors also exist (e.g. "jangedoo/utkface-new"). See --utkface.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DO_VERIFY=0
DO_UTK=0
for arg in "$@"; do
  case "$arg" in
    --verify) DO_VERIFY=1 ;;
    --utkface) DO_UTK=1 ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (use --verify, --utkface, or --help)" >&2
      exit 2
      ;;
  esac
done

RAW="data/raw"
mkdir -p "$RAW"

echo ">> Downloading tabular datasets into ${RAW}/ ..."
echo "   (Adult UCI, Credit UCI default-of-credit-card-clients, LSAC public mirror,"
echo "    COMPAS ProPublica two-year, German Credit UCI statlog)"
python3 - <<'PY'
from src.data.datasets import (
    load_adult, load_credit, load_lsac, load_compas, load_german,
)
for loader in (load_adult, load_credit, load_lsac, load_compas, load_german):
    try:
        X, y, a, name = loader(data_dir="data/raw")
        print(f"  OK: {loader.__name__} -> {name}  X={getattr(X, 'shape', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {loader.__name__}: {e}")
        raise SystemExit(1)
PY

# Expected SHA-256 of the raw files as shipped/used on 2026-08-04 (this machine).
# If a public host rewrites bytes, verification fails loudly — re-check provenance.
# These are integrity aids, not cryptographic trust in the remote.
EXPECTED_SHA256=$(cat <<'EOF'
5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d  data/raw/adult.data
a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05  data/raw/adult.test
30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933  data/raw/default_of_credit_card_clients.xls
76244ae957d224a9cc49464196f53ad621585705418ad3c3cda27a7699471a16  data/raw/lsac.csv
c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d  data/raw/compas-scores-two-years.csv
b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871  data/raw/german.data
EOF
)

if [[ "$DO_VERIFY" -eq 1 ]]; then
  echo
  echo ">> Verifying SHA-256 checksums of raw tabular files ..."
  if command -v shasum >/dev/null 2>&1; then
    HASH_CMD=(shasum -a 256)
  elif command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD=(sha256sum)
  else
    echo "  ERROR: need shasum or sha256sum for --verify" >&2
    exit 1
  fi
  while read -r exp path; do
    [[ -z "${exp:-}" ]] && continue
    if [[ ! -f "$path" ]]; then
      echo "  MISSING: $path" >&2
      exit 1
    fi
    got="$("${HASH_CMD[@]}" "$path" | awk '{print $1}')"
    if [[ "$got" != "$exp" ]]; then
      echo "  MISMATCH: $path" >&2
      echo "    expected: $exp" >&2
      echo "    got:      $got" >&2
      echo "  Re-download from the URLs in this script header, or update EXPECTED_SHA256 if the" >&2
      echo "  public mirror intentionally changed. Do not silently ignore." >&2
      exit 1
    fi
    echo "  OK: $path"
  done <<< "$EXPECTED_SHA256"
  echo ">> Checksums match expected (2026-08-04 pins)."
else
  echo
  echo ">> Tip: re-run with --verify to check SHA-256 against known pins."
fi

if [[ "$DO_UTK" -eq 1 ]]; then
  echo
  echo ">> UTKFace (manual; not auto-downloaded — archive is large):"
  echo "   1. Obtain aligned+cropped UTKFace images, e.g.:"
  echo "        - Official / historical Drive folder (search UTKFace aligned and cropped)"
  echo "        - Kaggle mirror: https://www.kaggle.com/datasets/jangedoo/utkface-new"
  echo "   2. Unpack so files match: data/raw/utkface/*.jpg.chip.jpg"
  echo "      (name pattern: {age}_{gender}_{race}_{date}.jpg.chip.jpg)"
  echo "   3. Extract ResNet18 features (MPS/CUDA recommended):"
  echo "        python3 scripts/extract_utkface_features.py"
  echo "   4. Run the image pipeline only with real features (never report synthetic"
  echo "      Gaussian smoke features as results). See docs/UTKFACE_PIPELINE.md."
  echo "   Note: data/raw/utkface_features_smoke.npz is a small smoke cache only."
else
  echo
  echo ">> Tabular data ready. Re-run with --utkface for UTKFace manual-fetch steps."
fi

echo ">> Done."
