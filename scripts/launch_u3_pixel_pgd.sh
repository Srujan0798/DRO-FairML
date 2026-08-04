#!/usr/bin/env bash
# Launch U3 pixel PGD on flair2 when a GPU is free. Does NOT kill U1/U2.
# Safe to run while U1 occupies one GPU — picks the other if truly idle.
# Run ON flair2 from /data/srujan.sai/DRO-FairML-run:
#   bash scripts/launch_u3_pixel_pgd.sh
set -euo pipefail
cd /data/srujan.sai/DRO-FairML-run

# 1) Images
if [[ ! -d /data/srujan.sai/UTKFace ]]; then
  echo "Linking images (CONFIRM=1)..."
  CONFIRM=1 bash scripts/flair2_link_utkface_images.sh
fi

# 2) Do not double-start U3
if pgrep -f 'experiments/run_utkface_pixel_pgd.py' >/dev/null 2>&1; then
  echo "U3 already running:"
  pgrep -af 'run_utkface_pixel_pgd' || true
  exit 0
fi

# 3) Pick a free GPU (idle: <200 MiB used — U1/U2 hold ~600+ MiB when alive)
#    May run alongside U1 if the other L40S is free (U2 finished).
free_gpu=""
while IFS=',' read -r idx mem; do
  idx=${idx// /}
  mem=${mem// /}
  mem=${mem//MiB/}
  if [[ -n "$idx" && "${mem%%.*}" -lt 200 ]]; then
    free_gpu=$idx
    break
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

if [[ -z "$free_gpu" ]]; then
  echo "No free GPU (all have >=200 MiB used). U1/U2 still holding devices:"
  pgrep -af 'run_utkface_server|run_utkface_multigroup|run_utkface_pixel' || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exit 1
fi

echo "Using GPU $free_gpu (concurrent U1 OK if on the other GPU)"
mkdir -p logs results
export CUDA_VISIBLE_DEVICES=$free_gpu
export PYTHONUNBUFFERED=1
nohup ./venv_gpu/bin/python -u experiments/run_utkface_pixel_pgd.py \
  --data_dir /data/srujan.sai/UTKFace \
  --n_seeds 6 --alphas 0.1 0.2 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 10 \
  --device cuda \
  --out results/utkface_pixel_pgd.json \
  > logs/u3_pixel_pgd.log 2>&1 &
echo "U3 PID $!  log logs/u3_pixel_pgd.log  out results/utkface_pixel_pgd.json"
sleep 2
head -30 logs/u3_pixel_pgd.log || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
