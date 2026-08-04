#!/usr/bin/env bash
# Launch U3 pixel PGD on flair2 when a GPU is free. Does NOT kill U1/U2.
# Run ON flair2 from /data/srujan.sai/DRO-FairML-run:
#   bash scripts/launch_u3_pixel_pgd.sh
set -euo pipefail
cd /data/srujan.sai/DRO-FairML-run

# 1) Images
if [[ ! -d /data/srujan.sai/UTKFace ]]; then
  echo "Linking images (CONFIRM=1)..."
  CONFIRM=1 bash scripts/flair2_link_utkface_images.sh
fi

# 2) Pick a free GPU (memory used < 1 GiB)
free_gpu=""
while read -r idx mem; do
  mem=${mem// MiB/}
  if [[ "${mem%%.*}" -lt 1000 ]]; then
    free_gpu=$idx
    break
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

if [[ -z "$free_gpu" ]]; then
  echo "No free GPU (all have >=1GiB used). U1/U2 still holding devices — wait."
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exit 1
fi

echo "Using GPU $free_gpu"
mkdir -p logs results
export CUDA_VISIBLE_DEVICES=$free_gpu
nohup ./venv_gpu/bin/python experiments/run_utkface_pixel_pgd.py \
  --data_dir /data/srujan.sai/UTKFace \
  --n_seeds 6 --alphas 0.1 0.2 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 10 \
  --device cuda \
  --out results/utkface_pixel_pgd.json \
  > logs/u3_pixel_pgd.log 2>&1 &
echo "U3 PID $!  log logs/u3_pixel_pgd.log  out results/utkface_pixel_pgd.json"
