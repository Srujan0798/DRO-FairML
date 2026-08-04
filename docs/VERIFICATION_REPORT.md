# Independent Verification Report (LIVE — 540 grid)

**Date:** 2026-08-04 (post-meeting cleanup)  
**Canonical:** `results/canonical_tau1.json`  
**Method:** Recompute means, win-counts, one-sided Wilcoxon (`naive DP > dro DP`) from raw rows.  
**Meeting brief:** `docs/MEETING_2026-08-04.md` (source of presentation numbers).

---

## 0. Grid status (LIVE)

| Attack | Rows | Status |
|--------|------|--------|
| `dp` | **180** | COMPLETE |
| `combined` | **180** | COMPLETE |
| `if` | **180** | COMPLETE |
| **Total** | **540** | COMPLETE (unique keys = 540) |

**Provenance (all 540 rows):** τ=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0, radii_mode=uniform, coordinated=False, seeds 0–5.

**IF non-degeneracy (attack=`if` only):** max |if_clean| ≈ **0.239** (all 180 rows ≫ 1e-6).

---

## 1. Ship-critical claims (recomputed LIVE)

### Adult / DP attack

| α | wins (DRO lower DP) | p (one-sided) | Note |
|---|---------------------|---------------|------|
| 0.0 | **6/6** | 0.0156 | |
| 0.1 | **5/6** | 0.0312 | seed 2 loses — **not 6/6 every α** |
| 0.2 | **6/6** | 0.0156 | |
| 0.3 | **6/6** | 0.0156 | high-α; below constant predictor |
| 0.4 | **6/6** | 0.0156 | high-α; below constant predictor |

Adult Naive accuracy (DP): α=0.3 → **0.6669**, α=0.4 → **0.5512** (DRO 0.6755 / 0.5607 — still below Adult baseline ~0.752).

### Adult / Combined
**6/6 every α**, p=0.0156.

### Credit / DP
**6/6 every α**, p=0.0156. Naive acc α=0.3/**0.4** = **0.7527 / 0.7513** (below Credit baseline ~0.779).

### Credit / Combined
α=0.1 is **5/6** (p=0.0312); others **6/6**.

### LSAC / DP
**0/6 every α** — degenerate. Accuracy **pinned ~0.902** (majority), **not** “below constant predictor.”

### LSAC / Combined
Wins at α=0.1/0.3/0.4 (p=0.0156); α=0.2 is 5/6.

### IF-attack third (honest)
See `results/if_wilcoxon_summary.txt`.

| Dataset | Verdict |
|---------|---------|
| Adult | MIXED — DP under IF wins α≤0.2; **LOSS α=0.3** (1/6); IF metric often improves |
| Credit | Mostly WIN on IF metric; DP under IF weaker at α=0.1 (4/6 n.s.) |
| LSAC | LOSS on DP under IF for α≤0.3 |

**Do not claim** a clean three-attack sweep on all datasets.

---

## 2. Mismatch ledger (closed)

| ID | Old error | Live status |
|----|-----------|-------------|
| M1 | Adult/DP “6/6 every α” | **MATCH** — α=0.1 is 5/6 |
| M2 | Naive acc 0.676/0.608 | **MATCH** — Naive 0.6669/0.5512 |
| M3 | Credit high-α wrong | **MATCH** — 0.7527/0.7513 |
| M4 | “every dataset below baseline” | **MATCH** — Adult+Credit only; LSAC pinned |
| M5 | n=3 τ=1 tables | **MATCH** — n=6 |
| M6 | “all three attacks” | **MATCH** — scoped; IF mixed |
| M7 | IF never generated / 0.0000 | **MATCH** — IF 180, max≈0.239 |

---

## 3. Figures (LIVE vs removed)

### Prefer for meeting / claims (regenerated 2026-08-04 from 540)
- `figures/fig_tau1_headline.pdf` — Adult DP wins **[6,5,6,6,6]**
- `figures/fig_final_wilcoxon_table.pdf` / `figD10_*`
- `figures/figD1`–`figD4` (constant predictor / tradeoff)
- Report suite `fig1/2/4/5/7`, `main_results`, `test_time_eval` (report includes)

### Removed (stale pre-540 or incomplete inputs)
- Jul-2 `fig_final_constant_*`, `fig_final_lambda_*`, `fig_final_tradeoff_*`
- Jul-20 `figD5`–`figD7` (need missing `results/individual/`)
- Jul-2 `fig_acc_win_curves_tau1.pdf`
- Optional high-α panels

---

## 4. UTKFace

`results/utkface_canonical.json` — **REAL** features / partial grid only.  
**No paper claim** until a reviewed multi-attack multi-seed subset exists.

---

## 5. How to re-verify

```bash
python3 -c "import json,collections;d=json.load(open('results/canonical_tau1.json'));print(len(d),dict(collections.Counter(r['attack'] for r in d)))"
# expect 540 and 180/180/180
python3 experiments/plot_meeting_figs_540.py
python3 -m pytest tests/ -q
make validate
```

**Ship judgment:** DP+Combined Adult/Credit α≤0.2 **PASS** (with 5/6 nuances). IF **MIXED**. LSAC/DP **document only**. UTKFace **not ready**.
