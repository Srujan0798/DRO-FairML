#!/usr/bin/env bash
# Quick GPU smoke on flair2 after unlock.
set -euo pipefail
REMOTE="${FLAIR2_REMOTE:-flair2}"
PROJ="${FLAIR2_REMOTE_PROJECT:-/data/srujan.sai/DRO-FairML-run}"
VENV=venv_gpu

ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
cd '$PROJ'
source ${VENV}/bin/activate
export PYTHONPATH='$PROJ'
python - <<'PY'
import torch, time
assert torch.cuda.is_available(), "CUDA not available"
print("devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
# matmul smoke
x = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
print(f"matmul_4096 ok in {time.time()-t0:.3f}s  sum={float(y.sum()):.3e}")
# DRO-Fair single config smoke (short epochs)
from experiments.run_fairness_pgd import run_single_experiment
t0 = time.time()
r = run_single_experiment(
    "credit", 0.2, 0, "dp", "dro",
    device="cuda", verbose=False,
    epochs=5, k_inner=3, pgd_steps=5, tau=1.0, n_seeds_planned=6,
)
print("dro_smoke", {k: r[k] for k in ("acc_clean","dp_clean","if_clean")},
      f"time={time.time()-t0:.1f}s")
print("GPU_SMOKE_OK")
PY
EOF
