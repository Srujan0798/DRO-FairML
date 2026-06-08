# Server Runbook — DRO-FairML Experiments

**For:** `flair2.iitgn.ac.in`  
**Project:** `/data/srujan.sai/DRO-FairML`  
**Data:** `/data/srujan.sai/UTKFace/` and `/data/srujan.sai/utkface_features.npz`

---

## Quick Start (copy-paste)

```bash
ssh flair2.iitgn.ac.in
cd /data/srujan.sai/DRO-FairML
git pull

# Option A: Sequential (safest, ~5-6 hours total)
bash experiments/run_all_server_experiments.sh

# Option B: Parallel (fast items concurrent, ~3-4 hours total)
bash experiments/run_all_server_parallel.sh

# After completion, generate report + figures
venv/bin/python3 experiments/aggregate_all_results.py
venv/bin/python3 experiments/generate_all_figures.py
venv/bin/python3 experiments/check_server_progress.py
```

---

## What Each Experiment Does

| # | Script | Time | GPU? | Output |
|---|--------|------|------|--------|
| 1 | `run_lambda_diagnostic.py` | ~10 min | No | `results/lambda_diagnostic.json` |
| 2 | `run_utkface_extended.py --mode lambda_max_cap` | ~20 min | Yes | `results/utkface_lambda_max_cap.json` |
| 5 | `run_utkface_extended.py --mode alpha_sweep` | ~20 min | Yes | `results/utkface_alpha_sweep.json` |
| 6 | `run_utkface_extended.py --mode fairness_pgd` | ~60 min | Yes | `results/utkface_fairness_pgd.json` |
| 3 | `run_utkface_pixel_pgd.py` | ~120 min | Yes | `results/utkface_pixel_pgd.json` |
| 4 | `run_utkface_randinit.py` | ~150 min | Yes | `results/utkface_randinit.json` |

**Total: ~400 min ≈ 6.5 hours** (sequential)

---

## Individual Commands

If you only want to run specific items:

### Item 2 — H3 test (most important)
```bash
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode lambda_max_cap --feature_cache /data/srujan.sai/utkface_features.npz \
    --n_seeds 5
```

### Item 5 — Alpha sweep {0.3, 0.4}
```bash
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode alpha_sweep --feature_cache /data/srujan.sai/utkface_features.npz \
    --n_seeds 5
```

### Item 6 — FairnessTargetedPGD on images
```bash
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode fairness_pgd --feature_cache /data/srujan.sai/utkface_features.npz \
    --n_seeds 5
```

### Item 3 — Pixel-space PGD (H2)
```bash
venv/bin/python3 experiments/run_utkface_pixel_pgd.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2
```

### Item 4 — Random-init ResNet18 (H1)
```bash
venv/bin/python3 experiments/run_utkface_randinit.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2
```

---

## Monitoring Progress

```bash
# Check which experiments have finished
venv/bin/python3 experiments/check_server_progress.py

# Watch a specific log in real-time
tail -f logs/server_batch_02_utkface_lmax.log
```

---

## If Feature Cache Is Missing

```bash
# Extract features once (takes ~45 min on L40S)
venv/bin/python3 scripts/extract_utkface_features.py \
    --data-dir /data/srujan.sai/UTKFace \
    --output /data/srujan.sai/utkface_features.npz
```

---

## Expected Results

### H3 prediction
If inner-max overshoot causes DRO inversion on UTKFace, then `lambda_max=0.5` should produce **lower DP violation** than `lambda_max=1.5`.

### H2 prediction
If feature-space attacks are unrealistic, then pixel-space PGD should produce **different DRO behavior** than feature-space attacks.

### H1 prediction
If ImageNet features lack demographic signal, then random-init ResNet18 should make DRO **stop inverting**.

---

## Post-Processing

After all experiments finish:

```bash
# Aggregate all results into one report
venv/bin/python3 experiments/aggregate_all_results.py

# Generate all publication figures
venv/bin/python3 experiments/generate_all_figures.py

# Check completeness
venv/bin/python3 experiments/check_server_progress.py
```

Outputs:
- `results/ALL_RESULTS_SUMMARY.md`
- `results/utkface_all_results.json`
- `figures/fig12_*.pdf` through `fig17_*.pdf`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Lower batch size in pixel_pgd or randinit scripts |
| `No UTKFace images found` | Run feature extraction script first |
| `git pull` conflicts | Stash local changes, pull, then re-apply |
| Process killed | Use `tmux` or `nohup` to keep sessions alive |

---

## tmux Cheat Sheet (recommended)

```bash
# Start detached session
tmux new-session -d -s dro_experiments 'bash experiments/run_all_server_parallel.sh'

# Attach to watch
tmux attach -t dro_experiments

# Detach: Ctrl+B then D
# List: tmux ls
# Kill: tmux kill-session -t dro_experiments
```
