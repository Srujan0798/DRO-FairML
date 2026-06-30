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

# ONE COMMAND runs everything (tabular re-run + all UTKFace + figures)
bash experiments/run_everything.sh

# Or with tmux (recommended — survives disconnect)
tmux new-session -d -s dro_full 'bash experiments/run_everything.sh'
tmux attach -t dro_full
# Detach: Ctrl+B then D
```

**Expected total time:** ~6-8 hours (tabular ~2-3h CPU, UTKFace ~3-4h GPU)

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

## UTKFace on flair2 (original May 19 task + current priority)

**Local proof (done)**: Canonical config smoke (K_inner=10, tau=1 fixed, provenance) → 2 rows in results/utkface_all_results.json "fairness_pgd" bucket.

**Server script (hardened)**: experiments/run_utkface_server.py now supports --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20, records full provenance, has canonical examples in --help.

**Ready commands** (copy-paste on flair2 after git pull):

```bash
# Full grid (recommended: one attack per invocation, 6 seeds)
nohup venv/bin/python3 experiments/run_utkface_server.py \
  --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  > logs/utkface_dp_server.log 2>&1 &

# tmux (strongly recommended)
tmux new-session -d -s utk_dp 'cd /data/srujan.sai/DRO-FairML && \
  venv/bin/python3 experiments/run_utkface_server.py \
    --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    > logs/utkface_dp_server.log 2>&1'

# Quick smoke (CPU, for validation)
python3 experiments/run_utkface_server.py --attack dp --n_seeds 1 --alphas 0.2 \
  --tau 1.0 --k_inner 3 --epochs 5 --pgd_steps 3 --device cpu
```

**Email to supin.gopi** (ready to send, full polished draft was archived during cleanup but the content above is what to include):
Use the commands + "local smoke proof with canonical config (K=10, tau=1)" + request for flair2 account + public key.

The original full draft + commands files were moved to docs/_archive during the June 16 structure cleanup to keep the root clean.

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

## Detailed flair2 copy-paste commands (consolidated from UTKFACE_SERVER_COMMANDS.txt)

# UTKFace Server Commands — Ready-to-Copy for flair2.iitgn.ac.in (GPU)
# =====================================================================
# Context (May 19 task from Madam): "email supin.gopi for account in flair2" + set up UTKFace on server.
# Local CPU smoke (canonical K_inner=10, tau=1 fixed, FairnessTargetedPGD=adversarial PGD, full provenance) COMPLETE.
#   Evidence: 2 rows in results/utkface_all_results.json["fairness_pgd"] (smoke-validated against current code).
#
# Server script: experiments/run_utkface_server.py  (now turnkey with --help, canonical defaults, provenance)
#   - Writes per-attack: results/utkface_{dp,if,combined}_server.json  (resume-safe)
#   - After runs: optionally aggregate via experiments/aggregate_all_results.py (or manual)
#
# Prerequisites on flair2 (once account granted):
#   ssh <user>@flair2.iitgn.ac.in
#   cd /data/srujan.sai/DRO-FairML
#   git pull
#   # ensure venv has torch+cuda + requirements; feature cache at /data/srujan.sai/utkface_features.npz (or will fallback)
#   mkdir -p logs results
#
# RECOMMENDED: use tmux (survives disconnect) or nohup.
# No heavy pixel attacks in this batch (use server script for feature-space FPGD only).
#
# ---------------------------------------------------------------------
# 1. CANONICAL FULL GRID (tau=1 fixed, K_inner=10, 6 seeds, all alphas 0.0-0.4)
#    Run one attack per session for clean logs/monitoring. Total ~3 attacks.
# ---------------------------------------------------------------------

# Attack: dp  (FairnessTargetedPGD targeting demographic parity)
tmux new-session -d -s utk_dp 'cd /data/srujan.sai/DRO-FairML && \
  venv/bin/python3 experiments/run_utkface_server.py \
    --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    > logs/utkface_dp_server.log 2>&1'
tmux attach -t utk_dp
# (detach: Ctrl+B D ; re-attach later: tmux attach -t utk_dp)

# Equiv nohup (fire-and-forget):
nohup venv/bin/python3 experiments/run_utkface_server.py \
  --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  > logs/utkface_dp_server.log 2>&1 &
echo $! > logs/utkface_dp_server.pid

# Attack: if
tmux new-session -d -s utk_if 'cd /data/srujan.sai/DRO-FairML && \
  venv/bin/python3 experiments/run_utkface_server.py \
    --attack if --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    > logs/utkface_if_server.log 2>&1'
# (attach + detach as above)

nohup venv/bin/python3 experiments/run_utkface_server.py \
  --attack if --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  > logs/utkface_if_server.log 2>&1 &
echo $! > logs/utkface_if_server.pid

# Attack: combined
tmux new-session -d -s utk_combined 'cd /data/srujan.sai/DRO-FairML && \
  venv/bin/python3 experiments/run_utkface_server.py \
    --attack combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    > logs/utkface_combined_server.log 2>&1'

nohup venv/bin/python3 experiments/run_utkface_server.py \
  --attack combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  > logs/utkface_combined_server.log 2>&1 &
echo $! > logs/utkface_combined_server.pid

# ---------------------------------------------------------------------
# 2. SINGLE-COMMAND ALL ATTACKS (simpler but one big log)
# ---------------------------------------------------------------------
nohup venv/bin/python3 experiments/run_utkface_server.py \
  --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  > logs/utkface_all_fpgd_server.log 2>&1 &
echo $! > logs/utkface_all_fpgd_server.pid

# tmux variant:
tmux new-session -d -s utk_fpgd 'cd /data/srujan.sai/DRO-FairML && \
  venv/bin/python3 experiments/run_utkface_server.py \
    --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    > logs/utkface_all_fpgd_server.log 2>&1'

# ---------------------------------------------------------------------
# 3. QUICK SMOKE / VALIDATION (CPU or 1-seed GPU sanity; matches existing 2-row smoke)
# ---------------------------------------------------------------------
python3 experiments/run_utkface_server.py \
  --attack dp --n_seeds 1 --alphas 0.2 --tau 1.0 --k_inner 3 --epochs 5 --pgd_steps 3 --device cpu

# ---------------------------------------------------------------------
# 4. SLURM-STYLE (if flair2 uses sbatch / srun; adapt queue/partition as needed)
# ---------------------------------------------------------------------
# cat > /tmp/utkface_fpgd.sbatch << 'SLURM'
# #!/bin/bash
# #SBATCH --job-name=utk_fpgd
# #SBATCH --output=logs/utkface_fpgd_%j.log
# #SBATCH --error=logs/utkface_fpgd_%j.err
# #SBATCH --time=04:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# cd /data/srujan.sai/DRO-FairML
# git pull
# venv/bin/python3 experiments/run_utkface_server.py \
#   --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
#   --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda
# SLURM
# sbatch /tmp/utkface_fpgd.sbatch

# ---------------------------------------------------------------------
# 5. MONITORING + POST-RUN (on flair2)
# ---------------------------------------------------------------------
# tail -f logs/utkface_dp_server.log
# ls -l results/utkface_*_server.json
# # After all attacks done, aggregate (updates utkface_all_results.json etc.):
# venv/bin/python3 experiments/aggregate_all_results.py
# # Or manual merge of the *_server.json into fairness_pgd bucket.
#
# Check provenance in results:
# python3 -c '
# import json
# for p in ["results/utkface_dp_server.json", "results/utkface_if_server.json"]:
#   if open(p):
#     r = json.load(open(p))[-1] if json.load(open(p)) else {}
#     print(p, {k:r.get(k) for k in ["k_inner","tau","pgd_steps","n_seeds_planned","epochs"]})
# '
#
# ---------------------------------------------------------------------
# 6. OPTIONAL: also run the extended modes if feature cache present (from SERVER_RUNBOOK)
#    (These are separate from the run_utkface_server.py focus here.)
# ---------------------------------------------------------------------
# venv/bin/python3 experiments/run_utkface_extended.py --mode fairness_pgd --feature_cache /data/srujan.sai/utkface_features.npz --n_seeds 6
# (But prefer the new server script above for canonical tau=1 + provenance.)

# ---------------------------------------------------------------------
# FILES UPDATED FOR THIS TASK:
# - experiments/run_utkface.py (added tau/k_inner/epochs/pgd_steps + provenance to single func + main --help)
# - experiments/run_utkface_server.py (turnkey argparse, canonical defaults n_seeds=6/tau=1/k=10, examples, provenance pass-through)
# - UTKFACE_SERVER_COMMANDS.txt (this file)
# - EMAIL_TO_SUPIN_GOPI_DRAFT.txt (polished)
#
# Evidence of local smoke validity (task 5): smoke rows have provenance keys + tau=1.0; current runner imports + schema compatible (quick structural check passed, no full retrain).
# =====================================================================
