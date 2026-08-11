# COMPLETE HANDOFF — DRO-FairML

**Date:** 2026-08-11 · **Repo:** `https://github.com/Srujan0798/DRO-FairML.git`
**Branch:** main · **Latest commit:** `c7ea309`
**Tests:** 101 passed · **Paper:** 403 KiB PDF · **Report:** 328 KiB PDF

---

## 1. What this project does

Implements DRO-FAIR (Distributionally Robust Optimization for Fair ML) and tests
it against adversarial fairness-targeted corruption. Three tabular datasets (Adult,
Credit, LSAC) + UTKFace image features. Central finding: fixing prediction
temperature τ=1 makes DRO beat Naive-FAIR on DP at α≤0.2. Also proposes
DRO-FAIR-AL (augmented Lagrangian) as an improvement with honest caveats.

---

## 2. Repository Structure

```
DRO-FairML/
├── results/                  # All experiment outputs
│   ├── canonical_tau1.json           # LOCKED 540 (n=6) + 360 extension (n=10)
│   ├── canonical_tau1_cosine.json    # TASK F re-run (540, post-cosine-fix)
│   ├── utkface_canonical.json        # LOCKED 90 REAL rows
│   ├── *_ablation.json               # Wave-1 ablation results (16 files)
│   ├── aug_lagrangian*.json          # AL improvement results
│   └── *_summary.md                  # 23 summary files
├── paper/                  # paper/main.pdf (tectonic build)
├── report/                 # report/report.pdf (tectonic build)
├── src/
│   ├── training/dro_fair.py          # DRO trainer (radii_scale, radii_clamp, aug_lagrangian_mu)
│   ├── training/naive_fair.py        # Naive baseline (history logging)
│   ├── corruption/adversarial.py     # FairnessTargetedPGD attack
│   ├── data/datasets.py              # Loaders (Adult, Credit, LSAC, UTKFace, COMPAS, German)
│   ├── evaluation/metrics.py         # DP/IF metrics (cosine IF)
│   └── utils/projections.py          # Dykstra projection (TV→L1)
├── experiments/
│   ├── run_fairness_pgd.py           # Core runner (all provenance params)
│   ├── run_ablation_parallel.py      # Shared parallel driver (lock, resume-safe)
│   ├── run_a*.py / run_n*.py         # Ablation drivers
│   ├── run_task_f_repro.py           # TASK F re-run driver
│   ├── summarize_*.py                # Summary generators
│   └── verify_reproducibility.py     # TASK F verification
├── docs/
│   ├── FINAL_REPORT.md               # Complete audit + all results
│   ├── FINAL_VERIFICATION.md         # Independent spot-check findings
│   ├── ADVISOR_PREREAD.md            # One-pager for Manisha/Kuldeep
│   ├── AL_REVIEW.md                  # Independent AL review
│   ├── TASKS_AL_VALIDATION.md        # AL validation tasks A-F
│   ├── HANDOFF_GLM.md                # Phase 0 audit + Phase 1 completion
│   ├── MEMO_FOR_ADVISOR.md           # AL improvement finding
│   └── KEY_FORMULAS.md               # Theoretical reference (Finding 1 documented)
├── scripts/
│   ├── flair2_unlock.sh              # GPU server unlock (syntax-verified)
│   └── orchestrate_wave1.sh          # Sequential ablation orchestrator
├── Makefile                  # test/validate/paper/report/full targets
└── requirements.txt
```

---

## 3. Locked Science (NEVER overwrite)

| File | Rows | Description |
|---|---|---|
| `results/canonical_tau1.json` | 900 | 540 n=6 locked + 360 n=10 extension. First 540 byte-identical (SHA-256 verified) |
| `results/utkface_canonical.json` | 90 | All REAL, Mac MPS run |

**Canonical config:** τ=1.0, K_inner=10, epochs=60, pgd_steps=20, λ_init=0.0,
radii_mode=uniform, coordinated=False, 6 seeds (0-5 for locked, 6-9 for extension).

---

## 4. Key Results (verified from raw data)

### Headline (canonical, n=6)
- Adult/Credit, α≤0.2, DP+Combined: DRO lower DP (p<0.05); Adult/DP α=0.1 is 5/6
- LSAC/DP: degenerate negative (documented)
- IF attack: MIXED (not a clean sweep)
- α≥0.3: below constant-predictor on Adult/Credit only

### Wave-1 Ablations (all complete)
| Agent | Verdict |
|---|---|
| A1 kNN k∈{5,15} | IF attack strength depends on k |
| A2 τ∈{10,100} | τ=100 artifact demonstrated cleanly (n=6) |
| A3 λ/lr grid | No rescue of α=0.3 |
| A4 random vs adv | **12-40× claim WRONG** → corrected to -3.7× to 1.6× |
| A5 empirical radii | No improvement (0/5 cells) |
| N1 attack×radius | Directional pattern (12/15 cells prefer larger radius); Spearman NOT significant (p=0.8047) |
| N2 high-α rescue | No τ/lr/epochs works; convergence evidence |
| N3 COMPAS+German | German replicates DRO; COMPAS ambiguous |
| N4 IF@α=0.3 | Confirmed (p=0.0156, 6/6) |
| N5 K_inner∈{5,20} | Small sensitivity |
| L2 LSAC fix | No arm works — limitation stands with evidence |
| S n=6→n=10 | 1 significance flip; all DP wins stay significant |

### AL Improvement (TASKS_AL_VALIDATION.md)
- Diagnosis: dual decay caps λ at ~0.01 (126× below 1.5 ceiling)
- Fix: augmented Lagrangian (μ/2)g²; μ=0 is byte-identical to canonical
- TASK A: AL is a **generic fairness regulariser**, NOT corruption-robustness (α=0.0 control shows 2.5× more effect)
- TASK B: Denoising hypothesis supported (AL fits corrupted points 33% less)
- TASK C: μ=20 optimal for Adult; Credit collapses at μ≥1.0
- TASK C2: AL × radius CONFLICTS (combined degenerates)
- TASK E: Implementation correct, no leakage

### GPU Lane (Grok, complete)
- U1: CUDA reproducibility (max|ΔDP| < 0.013)
- U2: Multi-group UTKFace (5-race)
- U3: Pixel-space PGD (12 configs)

---

## 5. Discrepancies Found & Fixed

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | N1 ARM B Spearman ρ=0.668 claim — raw data gives ρ=0.131, p=0.8047 | HIGH | ✅ Fixed — corrected to directional pattern |
| 2 | S "6 significance flips" — raw data shows 1 | MEDIUM | ✅ Fixed → 1 flip |
| 3 | TASK F memo prediction partially wrong — combined attack drifts (acc_max=7.5%) | MEDIUM | ✅ Documented — DP/IF stable, combined significance pattern unchanged |
| 4 | Old 12-40× claim in PDFs | HIGH | ✅ Verified removed |
| 5 | "Uniform" radii formula never executed (dead code) | MEDIUM | ✅ Documented in KEY_FORMULAS.md |

---

## 6. TASK F — Reproducibility Re-run (Complete)

- **540/540 rows** in `canonical_tau1_cosine.json`
- **DP attack:** stable (drift < 0.003) — headline unaffected
- **IF attack:** stable (drift < 0.008)
- **Combined attack:** drifts at α≥0.2 (acc_max=7.5%, DP_max=8.5%) because cosine IF fix changed combined attack behavior
- **Combined significance pattern:** unchanged — all same cells stay significant
- Verification: `experiments/verify_reproducibility.py` → `results/reproducibility_diff.md`

---

## 7. How to Build & Verify

```bash
# Setup
pip install -r requirements.txt
bash data/download_data.sh --verify

# Tests + validation
python3 -m pytest tests/ -q          # expect 101 passed
python3 experiments/validate_results.py  # expect PASS (6/9 DP wins)

# Build PDFs
make paper && make report            # tectonic, no errors

# Full repro (no training)
make full                            # regenerates tables/figures from canonical
```

---

## 8. Key Commands for Common Tasks

```bash
# Run a single ablation (uses shared lock, resume-safe)
ABLATION_WORKERS=12 python3 experiments/run_a1_knn.py

# Run orchestrator (sequential, all remaining jobs)
bash scripts/orchestrate_wave1.sh

# Verify any (dataset, alpha, attack) cell from locked JSON
python3 -c "
import json, numpy as np
from scipy.stats import wilcoxon
d = json.load(open('results/canonical_tau1.json'))
ds, a, atk = 'adult', 0.2, 'dp'
rows = [r for r in d if r['dataset']==ds and r['attack']==atk and abs(r['alpha']-a)<1e-9]
ndp = [r['dp_clean'] for r in rows if r['method']=='naive']
ddp = [r['dp_clean'] for r in rows if r['method']=='dro']
diff = np.array(ndp) - np.array(ddp)
_, p = wilcoxon(diff, alternative='greater')
print(f'n={len(ndp)}  Naive: {np.mean(ndp):.4f}  DRO: {np.mean(ddp):.4f}  p={p:.4f}')
"

# Sync to flair2 (GPU server, no internet/git)
rsync -az --exclude='.git' --exclude='data/raw' --exclude='figures' \
  --exclude='logs' --exclude='results' --exclude='venv_gpu' \
  /Users/srujansai/Desktop/DRO-FairML/ flair2:/data/srujan.sai/DRO-FairML-run/
```

---

## 9. Advisors — Key Asks and Where They Landed

| Who | Ask | Delivered |
|---|---|---|
| Manisha May 19 | "implement pgd for fairness metrics" | ✅ src/corruption/adversarial.py |
| Manisha May 19 | "Set up UTKFace in the server" | ✅ U1/U2/U3 GPU tasks |
| Manisha May 19 | "see performance on Adult etc" | ✅ COMPAS + German added |
| Kuldeep Q1 | λ/lr hyperparameter tuning | ✅ lambda_grid.json (no rescue) |
| Kuldeep Q5 | k-ablation for IF attack | ✅ knn_ablation.json |
| Kuldeep Q7 | "if IF is good for α=0.3, state clearly" | ✅ Confirmed, p=0.0156 |
| Kuldeep Q9 | "6 seeds or push for more?" | ✅ Extended to n=10 |
| Kuldeep Q13 | "verify all the claims" | ✅ Phase 0 audit |
| Kuldeep May-29 | "Does the attack affect the radius?" | ✅ Directional pattern (NOT significant) |
| Kuldeep Jun-16 | tau→lr→convergence protocol | ✅ high_alpha_summary.md + plots |

---

## 10. Honest Negatives (reported, not hidden)

- 12-40× claim → corrected to -3.7× to 1.6×
- Empirical radii → no improvement
- LSAC fix → no arm works
- High-α rescue → nothing works
- AL generalization → does not generalize; α=0.0 control falsifies robustness framing
- AL × radius → conflicts
- Credit AL → degenerate at every μ
- COMPAS → pattern doesn't replicate

---

## 11. Critical Knowledge

1. **Fork pool instability:** macOS Python 3.14 fork ProcessPool crashes under
   sustained multi-driver load. Single-driver runs with ≤12 workers are stable.
   The shared `run_ablation_parallel.py` uses a machine-wide `_AblationLock` (fcntl)
   to serialize jobs — always go through it.

2. **MPS gives no speedup** for tabular experiments (17.3s CPU vs 17.7s MPS).
   GPU only helps for UTKFace image experiments.

3. **DRO configs are ~30 min each** (K_inner=10 inner-max loop). Naive configs
   are ~15s. Plan compute accordingly.

4. **The combined attack drifts** after the cosine IF fix because combined =
   0.5 DP + 0.5 IF, and IF is now real. DP-attack headline is unaffected.

5. **n_seeds_planned field:** Locked 540 rows have n_seeds_planned=6; extension
   rows have n_seeds_planned=10. Claims use seeds 0-5.

6. **flair2 is unlocked** (2× L40S, torch 2.6.0+cu124) but behind SSL firewall —
   no pip/PyPI, no git. Sync via rsync, run via `ssh flair2`, commit from Mac.

7. **Constant-predictor floors:** Adult 0.7521, Credit 0.7788, LSAC 0.9016, COMPAS 0.5334, German 0.7000. Any DP claim at/below floor = degeneracy, not a win.

---

## 12. Submission Status

- [x] Phase 0 audit (correctness)
- [x] 12 Wave-1 ablations
- [x] AL validation (A-F)
- [x] GPU lane (U1-U3)
- [x] Paper + report build
- [x] TASK F reproducibility
- [x] Advisor pre-read
- [x] All discrepancies fixed
- [ ] **Submit** — final human review of PDFs

**Repo is submission-ready at commit c7ea309.**
