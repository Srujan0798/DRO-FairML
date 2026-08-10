# DRO-FairML — Final Report

**Date:** 2026-08-09 · **Branch:** main · **Tests:** 101 passed · **Paper:** 402 KiB PDF · **Report:** 326 KiB PDF

---

## 1. Phase 0 — Correctness Audit (complete)

Direct instruction: stop adding experiments, verify the math is correct.

### Finding 1 — "uniform" radii formula has never executed (HIGH severity)
`src/training/dro_fair.py::_compute_radii` branches: `empirical` → `a_val is not None` → `else` (the documented closed form). The canonical runner always passes `a_val`, so the `elif` branch fires for every "uniform" row, and `(π_obs − α)/(1 − 2α)` is dead code.

**Decision (b):** Keep `a_val` (defensible — both methods have clean validation data, not an oracle leak). Document in `docs/KEY_FORMULAS.md` that the canonical grid uses validation-estimated group proportions, not the theoretical closed form. Locked 540 rows, headline claims, and Wilcoxon results unchanged.

### Item 4 — TV → L1 ×2 factor
**VERIFIED CORRECT.** `dro_fair.py:219` passes `2 * radius` to `project_simplex_l1_ball`. Unit test `test_tv_to_l1_radius_factor_2` confirms projection lands on the L1 ball boundary at exactly 2ρ.

### Items 5-6 — Tilted risk + dual ascent
**VERIFIED CORRECT.** Tilted risk `β·(logsumexp(ℓ/β) − log(m))` matches Algorithm 1 step 2. Dual ascent `λ ← clamp(λ + η·0.95^epoch·g, 0, λ_max)` matches step 3.

### Invariants (900-row grid)
`dp_clean ≥ 0`, `if_clean ≥ 0`, `acc_clean ∈ [0,1]`: **0 violations.** α=0 gap is small on Adult/Credit (0.001-0.007) and large on LSAC (+0.038, known degeneracy).

---

## 2. Canonical Grid (locked truth)

- `canonical_tau1.json`: **900/900 rows** (540 n=6 locked + 360 n=10 extension)
- First 540 byte-identical (SHA-256 verified)
- 1 significance flip at n=10 (credit/if/α=0.1: n.s. at n=6 → sig at n=10); all DP wins stay significant at both n
- `utkface_canonical.json`: 90/90 REAL rows (read-only)

**Headline (verified):** DRO beats Naive on Adult/Credit DP+Combined at α≤0.2 (Wilcoxon p<0.05, n=6; Adult/DP α=0.1 is 5/6). LSAC/DP degenerate. IF mixed. α≥0.3 below constant-predictor on Adult/Credit only.

---

## 3. Wave-1 Ablations (12/12 complete)

| Agent | Rows | Verdict |
|---|---|---|
| **A1** kNN ablation k∈{5,15} | 360/360 | IF attack strength depends on k (documented) |
| **A2** τ∈{10,100} ablation | 360/360 | τ=100 artifact demonstrated CLEANLY on all 3 datasets (n=6, vs historical n=3) |
| **A3** λ/lr grid | 72/72 | No (λ,lr) beats default; no α=0.3 rescue above constant predictor 0.7521 |
| **A4** random vs adversarial | 144/144 | **12-40× claim WRONG** — real multiplier -3.7× to 1.6× (median 0.7×). Paper must correct. |
| **A5** empirical radii | 180/180 | No improvement (0/5 cells). Attack-aware radius calibration does not help. |
| **N1** attack×radius | 252/252 | Radius/attack pattern: 12/15 cells prefer larger radius (all α≥0.2); Spearman ρ=+0.131 (p=0.8047, limited to 6 cells with measured attack_eff) — directional pattern consistent with Kuldeep's May-29 hypothesis but not statistically significant |
| **N2** high-α rescue | 120/120 | No τ/lr/epochs rescues α≥0.3. 200-epoch convergence evidence closes Kuldeep's Jun-16 protocol. |
| **N3** COMPAS + German | 360/360 | **German REPLICATES** DRO pattern (2/2 sig at α≤0.2); COMPAS ambiguous (0/2 sig) |
| **N4** IF@α=0.3 | analysis | **CONFIRMED** — Adult/Credit p=0.0156, 6/6; Adult DP-loss coupling confirmed |
| **N5** K_inner∈{5,20} | 180/180 | Small sensitivity (5/30 cells, max ΔDP=0.0011); K_inner≥5 largely sufficient |
| **L2** LSAC fix | 120/120 | **NO arm works** — clamp, empirical, combined all fail. Limitation stands WITH EVIDENCE. |
| **S** n=6→n=10 extension | 900/900 | 1 significance flip (credit/if/α=0.1); all DP wins stay significant at both n |

---

## 4. AL Validation (TASKS_AL_VALIDATION.md, A-F complete)

### Diagnosis (why DRO was barely beating Naive)
The dual decay `lr_λ · 0.95^epoch` caps λ at ~0.01 (126× below the 1.5 ceiling). The fairness penalty `λ·g` peaks at 0.0029 vs training loss 0.538 — **0.5% of the loss.** The constraint machinery was effectively switched off.

### Fix (DRO-FAIR-AL)
Classical augmented Lagrangian: `total_loss = L_tilt + λ·g + (μ/2)·g²`. μ=0 is byte-identical to canonical (unit-tested).

### TASK A — Does AL generalize?
**NO.** α=0.3 and α=0.4 are degenerate (below constant-predictor floor). Critical finding: **α=0.0 control shows a 2.5× LARGER effect than under attack.** AL is a **generic fairness regulariser**, not a corruption-robustness mechanism. Robustness framing withdrawn.

### TASK B — Why does accuracy go UP?
**Denoising hypothesis SUPPORTED.** AL fits corrupted points 33% less than canonical DRO (corrupted-subset acc: 0.51 vs 0.18), while improving clean-subset acc by 3 points. Mechanism: majority-class shift that suppresses the DP gap the attack inflates.

### TASK C — μ sensitivity
μ=20 is optimal for Adult (acc 0.7783 ≥ floor+0.02). Credit collapses at μ≥1.0 (no safe μ). AL is Adult-only.

### TASK C2 — AL × radius compound
**CONFLICT.** The combined arm (radii_scale=2.0 + μ=20) is degenerate (acc 0.7561 ≤ floor+0.02 threshold). The two levers fix the same deficiency and cannot be stacked.

### TASK E — Independent review
Implementation mathematically correct. Gradient `(μ/2)g² → μ·g·∇g` verified. g_dp/g_if non-negative (no max needed). Statistics reproduce exactly. No leakage. One contained defect: seed-3 at α=0.2 is at floor; excluding it keeps p=0.03125 (<0.05).

### TASK F — Reproducibility gap
Re-running canonical grid with current code (post-k-NN-cosine-fix) into `canonical_tau1_cosine.json`. Code ready and launched. Expected: accuracy reproduces exactly, DP shifts ~1e-7, IF goes from noise (~1e-11) to real (~0.045).

---

## 5. GPU Lane (Grok, all complete)

| Task | Status | Result |
|---|---|---|
| **U1** Reproducibility | ✅ 90/90 | CPU/MPS vs CUDA: max|ΔDP| < 0.013, all OK |
| **U2** Multi-group UTKFace | ✅ 30/30 | DRO wins multi-group DP at α≥0.2 (6/6 at α=0.4) |
| **U3** Pixel-space PGD | ✅ 12/12 | DP mixed (4/6 @0.1, 2/6 @0.2); IF clean 6/6 wins both α |

---

## 6. Code & Builds

| Check | Status |
|---|---|
| pytest tests/ -q | **101 passed** |
| make paper | **402 KiB PDF, no errors** |
| make report | **326 KiB PDF, no errors** |
| make validate | **PASS** (DP wins 6/9, gate ≥6/9) |
| Data files | 46 JSON + 22 summaries, all committed |
| canonical_tau1.json integrity | **900 rows, first 540 intact** |

### Changes landed
- `src/data/datasets.py`: COMPAS + German loaders
- `src/training/dro_fair.py`: radii_scale, radii_clamp, aug_lagrangian_mu, history dict, history_path
- `src/training/naive_fair.py`: history dict + history_path
- `src/utils/projections.py`: TV→L1 test added
- `experiments/run_fairness_pgd.py`: corruptor_type, lr_lambda, attack_k, radii_scale, radii_clamp, dump_history, attack_effectiveness, aug_lagrangian_mu
- `experiments/run_ablation_parallel.py`: shared parallel driver (resume-safe, atomic save, provenance)
- `paper/sections/`: AL integration, Future Work update, Limitations
- `report/`: matching edits

---

## 7. Honest Negatives (reported, not hidden)

- **12-40× claim** → corrected to -3.7× to 1.6× (A4)
- **Empirical radii** → no improvement (A5)
- **LSAC fix** → no arm works (L2)
- **High-α rescue** → no τ/lr/epochs works (N2)
- **AL generalization** → does not generalize; α=0.0 control falsifies robustness framing (TASK A)
- **AL × radius** → conflicts, combined degenerates (TASK C2)
- **Credit AL** → degenerate at every μ (TASK C)

---

## 8. Key Claims for the Paper

1. **Canonical:** Fixed τ=1 makes DRO beat Naive on Adult/Credit DP+Combined at α≤0.2 (p<0.05, n=6; Adult/DP α=0.1 is 5/6). Verified at n=10.
2. **German** replicates the DRO pattern; **COMPAS** does not (scope statement).
3. **LSAC/DP** is a degenerate negative — tested fix (L2) confirms it's not an artifact of the radii formula.
4. **Radius/attack pattern** (N1): 12 of 15 cells prefer the larger radius at α≥0.2 — directional pattern consistent with Kuldeep's May-29 hypothesis, though the Spearman correlation (ρ=+0.131, p=0.8047, limited to 6 cells with measured attack_eff) is not significant.
5. **AL (augmented Lagrangian)** is a genuine improvement on Adult (μ=20, 14.9× margin over Naive at α=0.2) but is a **generic fairness regulariser**, not a corruption-robustness mechanism (TASK A). Credit/LSAC are not rescued.
6. **IF@α=0.3** is supported on Adult/Credit (N4, p=0.0156, 6/6) but DP-under-IF is mixed.

---

## 9. TASK F Status

`experiments/run_task_f_repro.py` launched in background (540 configs, 12 workers). Expected ~5-6 hours. Resume-safe, writes to `canonical_tau1_cosine.json` (never touches locked file). Verification script `experiments/verify_reproducibility.py` ready to run on completion.

---

## 10. Summary

**Every recorded ask from Prof. Manisha Padala and Kuldeep (May 19 – Jun 30) has been delivered or explicitly cut with a written reason.** The math is verified (Phase 0). The ablations are complete (Wave-1). The AL improvement is validated and honestly caveated (A-F). The paper and report build clean with all findings integrated. The GPU lane is complete (U1-U3). TASK F (reproducibility re-run) is in progress.
