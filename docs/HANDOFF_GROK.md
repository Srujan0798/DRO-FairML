# HANDOFF — Grok: flair2 GPU work

**Your lane: flair2 GPU only.** You do not touch the Mac CPU ablation queue (GLM owns
that — `results/knn_ablation.json`, `tau_ablation.json`, `lambda_grid.json`,
`random_vs_adversarial.json`, `empirical_radii.json`, `kinner_ablation.json`,
`attack_strength.json`, `radius_sensitivity.json`, `high_alpha_tau.json`,
`canonical_tau1.json`'s n=10 extension — none of those are yours). You do not touch
`paper/*.tex`, `report/*.tex`, or `STATUS.md` — integration is a separate later pass.

## flair2 is already unlocked — do not reinstall

```
torch 2.6.0+cu124
cuda_available: True
n_gpu: 2
gpu 0: NVIDIA L40S 48GB
gpu 1: NVIDIA L40S 48GB
```
`venv_gpu` exists at `/data/srujan.sai/DRO-FairML-run/venv_gpu`. Use it directly:
`./venv_gpu/bin/python ...`. **Do not download wheels again, do not recreate venv_gpu**
— that already happened twice tonight (two agents independently redid it) and wasted
~6GB and time for nothing. Verify it still works before starting anything new:
```bash
ssh flair2 '/data/srujan.sai/DRO-FairML-run/venv_gpu/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
```
Should print `True 2`. If it doesn't, stop and report — don't reinstall blind.

## Locked truth — read once, don't re-derive

- `results/utkface_canonical.json`: **90/90 REAL rows, complete, on Mac MPS** (dp/if/
  combined = 30/30/30, all 6 seeds, all 5 alphas). This is done. Your GPU work is either
  (a) reproducing it on CUDA for a reproducibility row, or (b) genuinely new experiments
  a laptop can't do. **Do not re-run the same grid pointlessly** — U1 below reproduces
  it deliberately for a specific reason (see below), that's the only repeat that's useful.
- `results/canonical_tau1.json`: 540-row locked tabular grid. Not your lane, read-only
  reference if needed.
- Central finding: fixed τ=1 makes DRO beat Naive on Adult/Credit DP+Combined at α≤0.2.
  LSAC/DP degenerate. IF MIXED. See `docs/LSAC_DEGENERACY.md`, `docs/VERIFICATION_REPORT.md`.

## Code sync — flair2 has no internet/git past the SSL firewall

Every session, before running anything, sync fresh code from the Mac:
```bash
rsync -az --exclude='.git' --exclude='data/raw' --exclude='figures' \
  --exclude='logs' --exclude='results' --exclude='venv_gpu' --exclude='paper' \
  --exclude='report' /Users/srujansai/Desktop/DRO-FairML/ \
  flair2:/data/srujan.sai/DRO-FairML-run/
```

## The 4 tasks, in priority order

Use `CUDA_VISIBLE_DEVICES=0` for one job, `=1` for another, so two can run at once
without colliding. flair2 is a **shared cluster** (other users have Jupyter kernels
running there) — check `nvidia-smi` before assuming both GPUs are free.

### U1 — Reproducibility (do this first, it's cheap and it's an advisor ask)
Run the identical UTKFace canonical protocol (τ=1, K=10, n=6, epochs=60, pgd_steps=20,
attacks dp/if/combined, α 0.0–0.4) but on CUDA instead of Mac MPS:
```bash
CUDA_VISIBLE_DEVICES=0 ./venv_gpu/bin/python experiments/run_utkface_server.py \
  --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  --output results/utkface_flair2.json
```
90 configs. Write `results/utkface_reproducibility_summary.md`: compare CPU/MPS (Mac)
vs CUDA numbers per (attack, alpha) — same seeds should be near-identical; large gaps
are a bug to report, not hide. This makes Manisha's original May-19 "set up UTKFace in
the server" ask literally, finally true.

### U2 — Multi-group UTKFace (5-race, not the binary collapse)
UTKFace race is currently binarized to White/non-White. `src/evaluation/metrics.py`
already computes max-min DP for >2 groups — wire the raw 5-category race label through
instead of the binary collapse (check the UTKFace loader for where that binarization
happens). Run: 5 alphas × 6 seeds × 2 methods, dp attack = 60 configs →
`results/utkface_multigroup.json`. Write `results/utkface_multigroup_summary.md`: does
DRO's advantage hold across all 5 groups, or concentrate/vanish for the smallest ones
(Indian/Other)? Likely the single most novel result available for the paper.

### U3 — Pixel-space PGD (the genuine GPU-only stretch)
Everything so far attacks cached ResNet18 features. This attacks raw pixels through the
network — recover the code from git history first:
```bash
git show 3c371a8:src/corruption/image_pgd.py > src/corruption/image_pgd.py
```
(commit 3c371a8 is the fixed version — check `git log --oneline --all -- src/corruption/image_pgd.py`
for the full history if you need the matching CNN classifier scaffolding from commit
a31d43f too.) Scope: α ∈ {0.1, 0.2}, 6 seeds, dp attack, 2 methods, on raw UTKFace JPEGs
(`data/raw/utkface/UTKFace/`, 23,708 real images, already on flair2) = 24 configs →
`results/utkface_pixel_pgd.json`. Write `results/pixel_pgd_summary.md` comparing
feature-space vs pixel-space attack strength. Timebox to ~half a day; report partial
honestly rather than block everything else.

### U4 — CelebA second modality (stretch, only if U1-U3 are done with time to spare)
Second independent image dataset, same canonical protocol. Cut without guilt if short
on time — U1-U3 alone are a strong result.

## Rules

- Full provenance on every row (tau, k_inner, epochs, pgd_steps, seed, n_seeds_planned,
  device='cuda', data_provenance='REAL' — never let synthetic UTKFace features get
  reported as real, the loader already rejects them, keep it that way).
- Never write `results/canonical_tau1.json` or the ORIGINAL `results/utkface_canonical.json`
  — always a new file.
- Rsync results back to the Mac and commit from there (flair2 has no git/PyPI access,
  it is not the source-of-truth repo, don't try to push from it).
- Do not edit `paper/*.tex` / `report/*.tex` — leave summaries ready for the integration pass.

## When done

Post status in STATUS.md's "what's left" section (append, don't rewrite the file) —
don't attempt final paper integration yourself.
