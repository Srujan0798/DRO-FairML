# UTKFace Status — Agent M

**Date:** 2026-08-04  
**Path taken:** **Path 2 — REAL data obtained and experiments running**  
**Machine:** Mac (Apple Silicon MPS)

---

## Path decision

| Path | Meaning | Chosen? |
|------|---------|---------|
| 1 | flair2 GPU server | Not available (no account) |
| **2** | **Public download + local MPS** | **YES** |
| 3 | Scope out / blocked | No |

---

## Data provenance (REAL)

| Item | Value |
|------|-------|
| Source | Kaggle dataset `jangedoo/utkface-new` |
| License note | copyright-authors (Kaggle listing) |
| Download cmd | `kaggle datasets download -d jangedoo/utkface-new -p data/raw/utkface --unzip` |
| Images path | `data/raw/utkface/UTKFace/` |
| Image count | **23,705** valid `{age}_{gender}_{race}_{date}.jpg.chip.jpg` |
| Feature cache | `data/raw/utkface_features.npz` |
| Backbone | ResNet18 ImageNet (`IMAGENET1K_V1`), pre-FC 512-d |
| Extract device | MPS (~63 s, ~381 img/s) |
| Extract script | `scripts/extract_utkface_features.py` |
| Meta tag in npz | `provenance=REAL_UTKFACE_IMAGES`, `synthetic=False` |

### Label / protected attribute (canonical task)

Trainers require **binary** protected attributes. Real setup:

- **y** = gender (0=Female, 1=Male)
- **a** = race binarized (**White=0 vs non-White=1**)

This replaces the old broken override `a := y` (which made fairness degenerate: A≡Y).  
Synthetic Gaussian fallback is **disabled by default** (`ALLOW_SYNTHETIC_UTKFACE=1` required for smoke).

### Race distribution (real imbalance — not synthetic)

White 10078 · Black 4526 · Asian 3434 · Indian 3975 · Others 1692  
Gender: Female 12391 · Male 11314

### What is NOT real

- `data/raw/utkface_features_smoke.npz` — 24k balanced-race features with near-Gaussian norms; **do not treat as real**
- Any historical rows with `dataset_display` / `dname` containing `synthetic`
- `docs/_archive/UTKFACE_RESULTS_SYNTHETIC_SMOKE_ONLY.md` — archived synthetic-only smoke

---

## Experiment protocol (canonical)

Matches tabular grid:

| Hyperparam | Value |
|------------|-------|
| tau | 1.0 (fixed) |
| k_inner | 10 |
| epochs | 60 |
| pgd_steps | 20 |
| n_seeds | 6 |
| alphas | 0.0, 0.1, 0.2, 0.3, 0.4 |
| attacks | dp, if, combined |
| lambda_max | 1.5 |
| device | mps |
| **Total configs** | **90** |

Output: `results/utkface_canonical.json`  
Runner: `experiments/run_utkface_server.py --output results/utkface_canonical.json`  
Log: `logs/utkface_canonical_run.log` · agent log: `logs/utkface_agent_m.log`

Every result row includes:

- `data_provenance`: `"REAL"`
- `dataset` / `dataset_display`: `"utkface"` / `"UTKFace"` (never synthetic)
- `label_def`: `gender`, `protected_def`: `race_binary`
- full provenance: `tau`, `k_inner`, `epochs`, `pgd_steps`, `n_seeds_planned`

---

## Progress

Updated live as the grid runs. Target: **90/90 REAL rows**.

| When (IST) | Rows | Notes |
|------------|------|-------|
| pre-CLEAR | 5–9 | paused / killed to protect IF sweep |
| post-CLEAR ~14:09 | running | pid `run_utkface_server.py` on MPS |
| **2026-08-04 ~14:17** | **12 / 90** | attack=`dp` only so far; α∈{0.0,0.1}; seeds 0–5; all `data_provenance=REAL` |
| **2026-08-04 ~14:25** | **16 / 90** | `dp` α∈{0.0,0.1,0.2} (α=0.2 still filling); all REAL |
| **2026-08-04 ~14:35** | **21 / 90** | `dp` only; α∈{0.0,0.1,0.2} complete (6 seeds); α=0.3 partial (3 seeds); all REAL |
| **2026-08-04 ~14:45** | **26 / 90** |
| **2026-08-04 ~14:46** | **27 / 90** | `dp` only; α∈{0.0–0.3} complete (6 seeds each); α=0.4 partial (2 seeds); all REAL |
| **2026-08-04 ~14:50** | **29 / 90** | `dp` only; α∈{0.0–0.3} complete; α=0.4 at 5/6 seeds; all REAL |

| **2026-08-04 ~14:51** | **34 / 90** | live: {'dp': 30, 'if': 4}; all REAL |

| **2026-08-04 ~14:55** | **38 / 90** | {'dp': 30, 'if': 8}; all REAL |

| **2026-08-04 ~15:00** | **41 / 90** | {'dp': 30, 'if': 11}; all REAL |

**Early REAL snapshot (α=0.0, attack=dp, 6 seeds, seed 0 shown):** Naive acc≈0.859, DP≈0.020, IF≈0.069; DRO acc≈0.859, DP≈0.020, IF≈0.052. Full tables only after 90/90.

**No paper claim until 90/90 (or scoped subset) verified.** Report real row count only.

Resume command if interrupted:

```bash
export PYTHONUNBUFFERED=1
python3 -u experiments/run_utkface_server.py \
  --attacks dp if combined \
  --alphas 0.0 0.1 0.2 0.3 0.4 \
  --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 \
  --device mps \
  --output results/utkface_canonical.json \
  | tee -a logs/utkface_canonical_run.log
```

---

## Early REAL numbers (α=0.0, attack=dp, 6 seeds)

Clean accuracy ≈ **0.86** for both Naive and DRO (ResNet features are predictive of gender).  
Clean DP ≈ **0.02** (small race disparity on gender prediction at α=0).  
All rows tagged `data_provenance=REAL`.

Full tables after grid completion — do not cite smoke/synthetic archive numbers.

---

## Code changes (Agent M)

1. `scripts/extract_utkface_features.py` — MPS/CUDA/CPU, provenance metadata in npz  
2. `src/data/datasets.py` — `load_utkface` auto-loads real cache; fails loudly if missing; binary race protected attr  
3. `experiments/run_utkface.py` — no synthetic by default; REAL tags; MPS; resume-safe writes; no A≡Y override  
4. `experiments/run_utkface_server.py` — device auto (cuda>mps>cpu); `--output` for single canonical JSON  

---

## Blockers

None for Path 2. Full grid ETA ~25–40 min on MPS after start (~17–27 s/config).
