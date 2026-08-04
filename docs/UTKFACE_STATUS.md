# UTKFace status (2026-08-04)

**Honest summary:** real image features are available and a **single-config MPS probe** completed with `data_provenance=REAL`. There is **no** multi-seed, multi-α, multi-attack UTKFace result set yet. **Do not** put UTKFace numbers in the paper or meeting “claims” section until a full protocol run exists.

## What exists on disk

| Item | Path / note |
|------|-------------|
| Images | `data/raw/utkface/` (Kaggle `utkface-new`, ~331MB zip extracted) |
| Features | `data/raw/utkface_features.npz` — X=(23705, 512), y=gender, a=race_binary (White/nonWhite), meta=REAL_UTKFACE_IMAGES (mtime ~13:44 IST) |
| Timing probe | `results/utkface_timing_probe.json` |
| Synthetic archive | `docs/_archive/UTKFACE_RESULTS_SYNTHETIC_SMOKE_ONLY.md` (old smoke tests only) |

## Probe result (not a full experiment)

Command (Agent M style):

```bash
python3 experiments/run_utkface.py --attack dp --alphas 0.0 --n_seeds 1 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 \
  --output results/utkface_timing_probe.json
```

| Field | Value |
|-------|--------|
| device | **mps** |
| attack / α / seed | dp / **0.0** / **0** |
| provenance | **REAL** |
| wall time | ~24 s (train+eval both methods) |
| Naive clean | acc≈0.859, dp≈0.020, if≈0.069 |
| DRO clean | acc≈0.859, dp≈0.020, if≈0.052 |

α=0 means clean==corrupted in the probe JSON (expected). This only shows the pipeline loads real features and trains; **no robustness claim**.

## What is still missing

- Full protocol: attacks ∈ {dp, combined, if} × α ∈ {0,0.1,0.2,0.3,0.4} × seeds 0–5 (or agreed subset).
- Wilcoxon / tables / paper language for image modality.
- Decision: ship as appendix modality **or** drop from Aug 10 scope if time-constrained.

## Sequencing

While tabular IF sweep (`run_if_parallel.py`, pid 10146) needs cores, **do not** thrash CPU with a full UTKFace grid. After **total=540**, push hard on MPS.

Rough cost from probe: ~24 s/config → e.g. 3×5×6 = 90 configs ≈ **~36 min** sequential (optimistic; IF attack may be slower).

## Claims policy

- ✅ “Real UTKFace features extracted; pipeline runs on MPS.”
- ❌ “DRO wins on UTKFace” / any multi-α table / inversion claims (historical synthetic inversion **withdrawn**).
