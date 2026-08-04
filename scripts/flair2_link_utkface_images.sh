#!/usr/bin/env bash
# flair2 only — prepare a local UTKFace JPEG tree for U3 (pixel PGD) WITHOUT re-download.
#
# Finds a world-readable UTKFace dir (prefer own copy; else known shared trees) and
# symlinks it to /data/srujan.sai/UTKFace and DRO-FairML-run/data/raw/utkface/UTKFace.
#
# Safe defaults: no copy of 23k JPEGs; symlink only. Does not start training jobs.
# Do not run while unsure about using another user's data — default requires CONFIRM=1.
set -euo pipefail

OWN="${OWN_UTKFACE:-/data/srujan.sai/UTKFace}"
RUN_LINK="${RUN_LINK:-/data/srujan.sai/DRO-FairML-run/data/raw/utkface/UTKFace}"
CANDIDATES=(
  "/data/srujan.sai/UTKFace"
  "/data/kshitish.madbhavi/UTKFace"
  "/data/kshitish.madbhavi/utkface_aligned_cropped/UTKFace"
  "/data/kshitish.madbhavi/kshitish/fl_fairness/data/UTKFace"
)

pick=""
for c in "${CANDIDATES[@]}"; do
  if [[ -d "$c" ]] && ls "$c"/*.jpg >/dev/null 2>&1; then
    n=$(find "$c" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
    echo "candidate $c n_jpg≈$n"
    if [[ -z "$pick" && "$n" -gt 1000 ]]; then
      pick=$c
    fi
  fi
done

if [[ -z "$pick" ]]; then
  echo "ERROR: no readable UTKFace JPEG tree found" >&2
  exit 1
fi

echo "selected source: $pick"
if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "Dry-run only. Re-run with CONFIRM=1 to create symlinks:"
  echo "  CONFIRM=1 bash scripts/flair2_link_utkface_images.sh"
  echo "Would link:"
  echo "  $OWN -> $pick  (if $OWN missing)"
  echo "  $RUN_LINK -> $pick"
  exit 0
fi

if [[ ! -e "$OWN" ]]; then
  ln -s "$pick" "$OWN"
  echo "linked $OWN -> $pick"
else
  echo "keep existing $OWN"
fi

mkdir -p "$(dirname "$RUN_LINK")"
if [[ -L "$RUN_LINK" || ! -e "$RUN_LINK" ]]; then
  rm -f "$RUN_LINK"
  ln -s "$pick" "$RUN_LINK"
  echo "linked $RUN_LINK -> $pick"
else
  echo "WARN: $RUN_LINK exists and is not a symlink; leave untouched" >&2
fi

echo "sample:"; ls "$RUN_LINK" | head -3
echo "DONE — U3 can point --image_dir at $RUN_LINK when a GPU is free (after U1/U2)."
