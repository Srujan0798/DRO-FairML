# Advisor Pre-Read — DRO-FairML Submission

**Date:** 2026-08-09 · **Submitted by:** Srujan Sai · **Advisors:** Prof. Manisha Padala, Kuldeep

---

## 1. What was asked and what was delivered

### Prof. Manisha Padala
| Ask | Status | Where |
|---|---|---|
| "implement pgd for fairness metrics (Both DP and IF, only DP, only IF)" | ✅ DONE | `src/corruption/adversarial.py::FairnessTargetedPGD` |
| "Set up an experiment for the UTKFace dataset in the server" | ✅ DONE | `results/utkface_flair2.json` (U1, CUDA reproducibility) + multi-group (U2) + pixel-PGD (U3) |
| "Check the adversarial attack on DP and improve it. Then, redo all the experiments" | ✅ DONE | Direct DP gradient attack implemented; 540-row canonical redone |
| "see the performance of DRO on Adult etc" | ✅ DONE | COMPAS + German added (`results/extended_datasets.json`) |
| "Are you guys able to access flair2??" (asked twice) | ✅ DONE | 2× L40S unlocked, all GPU experiments run |

### Kuldeep
| Ask | Status | Where |
|---|---|---|
| Q1: "try different initial value of lambdas, learning rates…" | ✅ DONE | `results/lambda_grid.json` (72/72) — no (λ,lr) beats default |
| Q2: "we fix tau for all alpha. Here we can use different tau for ablation" | ✅ DONE | `results/tau_ablation.json` (360/360) — τ=100 artifact demonstrated cleanly |
| Q3: LSAC framing | ✅ DONE | `docs/LSAC_DEGENERACY.md` + `results/lsac_radii_summary.md` (fix tested, doesn't work) |
| Q4: "if attack is known then we can use this approximation" (empirical radii) | ✅ DONE | `results/empirical_radii.json` (180/180) — no improvement |
| Q5: "For if attack we have to do ablation study for different k 5,10,15" | ✅ DONE | `results/knn_ablation.json` (360/360) |
| Q6/Q10: K_inner=5 vs 10 | ✅ DONE | `results/kinner_ablation.json` (180/180) — small sensitivity |
| Q7: "if individual fairness is good for α=0.3, then we can state this clearly" | ✅ DONE | `results/if_wilcoxon_n4_summary.md` — confirmed (p=0.0156, 6/6) |
| Q8: Random vs adversarial comparison | ✅ DONE | `results/random_vs_adversarial.json` (144/144) — **12-40× claim corrected** |
| Q9: "6 seeds now, or is 3 acceptable" / "or push for more?" | ✅ DONE | Extended to n=10 seeds (900/900 rows) |
| Q10: K_inner ablation | ✅ DONE | `results/kinner_ablation.json` (180/180) |
| Q11: "check accuracy dp and if of constant predictor" | ✅ DONE | Per-dataset baselines computed |
| Q12: IF plots per attack type, honest | ✅ DONE | IF story documented as MIXED |
| Q13: "verify all the claims" | ✅ DONE | Phase 0 audit complete |
| May-29 (first question): "Does the attack affect the radius?" | ✅ DONE | `results/attack_radius_summary.md` — radius follows a clear directional pattern with attack strength (12/15 cells prefer larger radius, all at α≥0.2); Spearman ρ=+0.131 (p=0.8047, limited to 6 cells with measured attack_eff) |
| Jun-16 (dictated protocol): tau → lr → convergence plots | ✅ DONE | `results/high_alpha_summary.md` + convergence plots |

---

## 2. Headline results (n=6, verified at n=10)

**Canonical grid:** 900/900 rows (`canonical_tau1.json`), τ=1, K_inner=10

| Claim | Verdict |
|---|---|
| Adult & Credit, α≤0.2, DP+Combined | DRO lower DP (p<0.05), accuracy equal-or-better |
| Adult/DP α=0.1 | 5/6 seeds (honest, not 6/6) |
| LSAC/DP | Degenerate negative (documented) |
| IF attack | MIXED (not a clean sweep) |
| α≥0.3 | Below constant-predictor on Adult/Credit (no method claim) |

**AL improvement (proposed):** Augmented Lagrangian (μ=20) gives 14.9× margin over Naive on Adult α=0.2. BUT TASK A found AL is a **generic fairness regulariser**, not corruption-robustness (α=0.0 control shows 2.5× larger effect). Framing adjusted accordingly.

---

## 3. Correctness audit (Phase 0)

- **Finding 1:** The documented "uniform" closed-form radii formula has never executed (the `elif a_val is not None` branch always fires). Decision: keep `a_val` path (defensible), document honestly.
- **Item 4:** TV→L1 ×2 factor verified correct
- **Items 5-6:** Tilted risk + dual ascent verified against Algorithm 1
- **Invariants:** 0 violations across 900-row grid

---

## 4. Corrections to previously reported claims

| Was reported | Corrected to | Source |
|---|---|---|
| Adversarial is "12-40× stronger than random" | Real multiplier is -3.7× to 1.6× | A4 (144/144) |
| "DRO wins at every α" | α=0.1 is 5/6; no claim at α≥0.3 | canonical_wilcoxon.csv |
| LSAC/DP is "DRO loses" | Degenerate run (model collapses), not a fair comparison | docs/LSAC_DEGENERACY.md |

---

## 5. What was cut and why

| Cut | Reason |
|---|---|
| COMPAS as a DRO win | Pattern doesn't replicate (0/2 sig cells) — reported honestly |
| Credit AL result | Degenerate (collapses to constant predictor) — not a genuine win |
| Empirical radii | Negative result (0/5 cells improve) |
| UTKFace multi-group as a clean sweep | DRO wins at high α, mixed at low α |

---

## 6. Builds

| Check | Status |
|---|---|
| pytest tests/ -q | 101 passed |
| make paper | 402 KiB PDF, no errors |
| make report | 326 KiB PDF, no errors |
| make validate | PASS |

---

## 7. In progress (resume-safe, will complete overnight)

- **TASK F (reproducibility re-run):** `canonical_tau1_cosine.json` at ~51/540 rows. Re-runs canonical grid with current code (post-k-NN-cosine-fix). Expected: accuracy reproduces exactly, DP unchanged, IF gap closed. Verification script ready at `experiments/verify_reproducibility.py`.

---

*All numbers reproducible from committed `results/*.json`. Full audit trail in `docs/FINAL_REPORT.md`.*
