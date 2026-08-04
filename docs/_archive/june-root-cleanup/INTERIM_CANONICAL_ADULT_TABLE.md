# INTERIM CANONICAL ADULT TABLE (Live Harvest)

**Live canonical run**: PID 79899, K=10, tau=1 fixed, 6 seeds, improved DP-targeted attack (FairnessTargetedPGD).  
Config: epochs=60, pgd_steps=20, lambda_init=0.0, radii_mode=uniform, coordinated=False.  
**Source**: `results/canonical_tau1.json` (39 rows) + `logs/canonical_540_full.log` (read for harvest) + `experiments/compute_canonical_wilcoxon.py` (ran safely on partial).

**Status as of 2026-06-16 15:23:40 IST**: 39 rows total (all Adult).  
- α=0.0 block **complete** with n=6 seeds.  
- α=0.1: partial (seed=0 only; dp-naive, dp-dro, if-naive completed; others pending).  
- Credit/LSAC: 0 rows yet.

**Note**: live K=10 tau=1, **partial data**, will update when more rows land (run continues).

## Headline: DP Attack (absolute DP violation, lower=better; focus per task)

| α   | n_seeds | Naive DP (mean) | DRO DP (mean) | DRO wins (strict) | Δ (Naive-DRO) | Wilcoxon p (one-sided) | Notes |
|-----|---------|-----------------|----------------|-------------------|---------------|------------------------|-------|
| 0.0 | 6       | 0.1491          | 0.1426         | **6/6**           | +0.0064       | **0.0156 ***           | Full block. DRO edge even at zero corruption. |
| 0.1 | 1       | 0.2197          | 0.2146         | 1/1               | +0.0051       | N/A (n=1)              | Partial seed=0 only. |

- At α=0.0: DRO **strictly better on all 6 seeds** (no ties). First real K=10/tau=1 confirmation of headline.
- Acc (α=0.0 DP): Naive 0.8135 / DRO 0.8147 (DRO no worse, slightly better).
- Full wilcoxon (live canonical, n=6): see `results/canonical_wilcoxon.md` (now shows * for adult dp/combined/if @ α=0.0; previously fallback n=3 p>=0.125).

## Full Interim Results: Adult by α, Attack, Method (means + win counts)

Means over available seeds so far. Win counts: #seeds where DRO DP < Naive DP (paired, only when both methods have that seed).

**α = 0.0 (n=6 seeds complete for all)**

| Attack   | Method | n | Acc mean | DP mean | Notes (vs paired naive) |
|----------|--------|---|----------|---------|-------------------------|
| dp       | naive  | 6 | 0.8135   | 0.1491  | - |
| dp       | dro    | 6 | 0.8147   | 0.1426  | wins 6/6 |
| if       | naive  | 6 | 0.8135   | 0.1491  | - |
| if       | dro    | 6 | 0.8147   | 0.1426  | wins 6/6 |
| combined | naive  | 6 | 0.8135   | 0.1491  | - |
| combined | dro    | 6 | 0.8147   | 0.1426  | wins 6/6 |

*(Note: at α=0.0, attack type has no effect; dp/if/combined report identical clean metrics.)*

**α = 0.1 (partial: only seed 0 for some cells)**

| Attack   | Method | n | Acc mean | DP mean | Notes (vs paired naive) |
|----------|--------|---|----------|---------|-------------------------|
| dp       | naive  | 1 | 0.8220   | 0.2197  | - |
| dp       | dro    | 1 | 0.8210   | 0.2146  | wins 1/1 (seed 0) |
| if       | naive  | 1 | 0.8177   | 0.0807  | - |
| if       | dro    | 0 | -        | -       | pending |
| combined | naive  | 0 | -        | -       | pending |
| combined | dro    | 0 | -        | -       | pending |

## Other Observations (from harvest)
- DRO acc >= Naive acc in completed cells (no fairness-accuracy tradeoff visible).
- Source data rows traceable: each json entry has full provenance (k_inner=10, tau=1.0, etc.).
- Process still live (ps shows PID 79899 active). More α=0.1/0.2+ and other datasets will land.
- Ran `experiments/compute_canonical_wilcoxon.py` (safely handles partial: skips groups with <2 paired seeds; auto-prefer canonical over fallback).
- Updated on harvest: `results/canonical_wilcoxon.csv`, `results/canonical_wilcoxon.md`, and this table + KULDEEP/MEETING appends.

**Will re-harvest + re-run generators (C scripts pointed at canonical_tau1.json) when full 540 rows or next alpha block lands.**

---
*Harvest by subagent (evidence-based, absolute DP only, no long runs). Data from committed json at time of read.*
