# flair2 NVIDIA GPU — READY

**Date:** 2026-08-04  
**Host:** `flair2` / `10.0.62.234` (`srujan.sai`)  
**GPUs:** 2× NVIDIA L40S 46GB (driver 570)

## Gate (verified)
```
torch 2.6.0+cu124
cuda True count 2
0 NVIDIA L40S
1 NVIDIA L40S
matmul_ok
```

## Paths
| What | Path |
|------|------|
| Project | `/data/srujan.sai/DRO-FairML-run` |
| Venv | `/data/srujan.sai/DRO-FairML-run/venv_gpu` |
| Wheelhouse | `/data/srujan.sai/wheelhouse` (~3.7G offline) |

## Use
```bash
ssh flair2
cd /data/srujan.sai/DRO-FairML-run
source venv_gpu/bin/activate
export PYTHONPATH=$PWD
# example
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# run experiment with device=cuda
python experiments/run_fairness_pgd.py --smoke  # edit to pass device if needed
```

## Notes
- flair2 has **no outbound PyPI** (SSL MITM). Always install offline from `wheelhouse/`.
- Mac local runs remain CPU/MPS. Canonical 540 + UTKFace 90 were **not** produced on flair2.
- To re-sync code: `rsync -az --exclude wheelhouse --exclude .git ./ flair2:/data/srujan.sai/DRO-FairML-run/`
