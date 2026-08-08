# FINAL VERIFICATION — Independent Claim Check

**Date:** 2026-08-09 · **Verifier:** independent recompute from raw JSON
**Scope:** docs/FINAL_REPORT.md's claims, recomputed from committed `results/*.json`

---

## 1. TASK F — Canonical reproducibility re-run

**Status:** IN PROGRESS (72/540 rows at time of writing, 13 workers alive).
`experiments/run_task_f_repro.py` writing to `results/canonical_tau1_cosine.json`.
Not yet complete — cannot verify the full diff. Prediction (from memo): accuracy
reproduces exactly, DP shifts ~1e-7, IF goes from noise to real.

---

## 2. Spot-check: N1 (attack × radius match)

**Report claim:** "ARM B: the radius that minimizes DRO DP grows significantly with
attack strength (Spearman ρ=+0.668, p=0.0065). 11 of 15 cells prefer radius 2.0."

**Recomputation from raw data:**

The report's ARM B analysis computes, per (dataset, α), the `best_radii_scale`
(the scale ∈ {0.5, 2.0} giving lowest mean DRO DP), then correlates
`best_radii_scale` with `attack_effectiveness` across 15 cells.

| Check | Report | Raw data | Match? |
|---|---|---|---|
| Best-radius counts | 11 of 15 prefer 2.0 | **12 of 15** prefer 2.0 | ⚠️ CLOSE — differs by 1 (tie at adult α=0.0 broken differently) |
| Spearman ρ | +0.668 | **+0.131** (6 cells with attack_eff) | ❌ **MAJOR DISCREPANCY** |
| p-value | 0.0065 | **0.8047** (6 cells) | ❌ **NOT SIGNIFICANT** |

**Root cause:** `attack_effectiveness` exists in `results/attack_strength.json`
only for α ∈ {0.1, 0.2} (6 cells). The report's correlation uses all 15 cells,
implying attack_eff values for α ∈ {0.0, 0.3, 0.4} that are NOT in any committed
file. The ρ=+0.668, p=0.0065 claim **cannot be reproduced from committed data**.

When computed from the 6 cells that DO have attack_eff, the correlation is
ρ=+0.131, p=0.8047 — not significant. The claim that "DRO's fairness is a
function of the radius/attack match" is **not supported by the data in the
committed result files**.

**Severity:** HIGH — this is a headline finding cited as answering Kuldeep's
original May-29 question. The paper's report PDF states ρ=0.6689.

**Note:** The per-cell best-radii-scale table in the report IS consistent with
the raw data (the 12 vs 11 count is a tie-breaking difference at adult α=0.0
where both scales give identical DP). The Spearman correlation is the problem.

---

## 3. Spot-check: A4 (random vs adversarial multiplier)

**Report claim:** "12-40× claim WRONG — corrected to 0.2-1.6×, median 0.7×."

**Recomputation from raw data** (`results/random_vs_adversarial.json`, 144 rows):

| Cell | DeltaDP_adv | DeltaDP_rnd | Multiplier |
|---|---|---|---|
| adult α=0.1 | +0.0027 | +0.0124 | 0.21× |
| adult α=0.2 | +0.0118 | +0.0106 | 1.11× |
| credit α=0.1 | +0.0016 | +0.0010 | 1.57× |
| credit α=0.2 | +0.0019 | +0.0017 | 1.13× |
| lsac α=0.1 | −0.0347 | +0.0093 | −3.72× |
| lsac α=0.2 | −0.0406 | +0.0126 | −3.21× |

**Range: −3.72× to 1.57×, median 0.66×.**

| Check | Report | Raw data | Match? |
|---|---|---|---|
| Range | −3.7× to 1.6× | −3.72× to 1.57× | ✅ YES |
| Median | 0.7× | 0.66× | ✅ YES (rounding) |
| 12-40× claim withdrawn | stated | confirmed | ✅ YES |

**PDF check:** Both paper and report contain the corrected range (−3.7× to 1.6×).
The old 12-40× number does NOT appear in either PDF. ✅

---

## 4. Spot-check: S (n=6→n=10 extension)

**Report claim:** "6 significance flips, all DP wins stay significant."

**Recomputation from raw data** (`canonical_tau1.json`, 900 rows):

| Metric | n=6 sig cells | n=10 sig cells |
|---|---|---|
| DP (attack) | 10 | 10 |
| IF (attack) | 7 | 8 |

**Significance flips (sig at one n, not sig at the other): 1**

| Cell | n=6 | n=10 |
|---|---|---|
| credit, if, α=0.1 | p=0.1094 (n.s.) | p=0.0098 (sig) |

| Check | Report | Raw data | Match? |
|---|---|---|---|
| DP wins stay significant | yes | yes (0 DP flips) | ✅ YES |
| Number of flips | **6** | **1** | ❌ **DISCREPANCY** |

**The "6 flips" claim does not match the raw data.** Only 1 cell changes
significance status between n=6 and n=10. All DP wins stay significant at both n
— this part is correct. The "6" may refer to cells where the p-value changed
substantially even if significance status didn't change, but by the standard
definition of a significance flip (sig → n.s. or n.s. → sig), the count is 1.

**Severity:** MEDIUM — the headline "all DP wins stay significant" is correct and
is the important claim. The "6 flips" is a secondary detail that doesn't affect
the conclusion but is numerically inaccurate.

---

## 5. Paper and report PDF verification

| Check | Status |
|---|---|
| `make paper` | ✅ builds (393 KiB, no errors) |
| `make report` | ✅ builds (318 KiB, no errors) |
| A4 corrected numbers in paper | ✅ present |
| A4 corrected numbers in report | ✅ present |
| Old 12-40× in paper | ❌ NOT FOUND (correctly removed) |
| Old 12-40× in report | ❌ NOT FOUND (correctly removed) |
| N1 ρ=0.668 in paper | ❌ NOT FOUND in paper (only in report) |
| N1 ρ=0.668 in report | ✅ present (but see discrepancy above) |

---

## 6. What I could not verify

- **TASK F:** Still running (72/540). Cannot verify the reproducibility diff until
  it completes. Run `python3 experiments/verify_reproducibility.py` when it finishes.
- **Other ablations (A1, A2, A3, A5, L2, N2, N3, N4):** Not independently
  recomputed, but their summary `.md` files are consistent with the raw data I
  spot-checked.

---

## 7. Summary

| Finding | Severity | Action needed |
|---|---|---|
| N1 ARM B Spearman ρ=+0.668, p=0.0065 cannot be reproduced from committed data | **HIGH** | Either add the missing attack_eff data for α ∈ {0.0, 0.3, 0.4}, or qualify/remove the claim |
| S "6 significance flips" — raw data shows 1 | **MEDIUM** | Correct to "1 flip" or clarify the definition |
| A4 corrected range | ✅ | No action — verified |
| S "all DP wins stay significant" | ✅ | No action — verified |
| TASK F | ⏳ | Run verification script when complete |
| Paper/report builds | ✅ | No action |
| Old 12-40× removed from PDFs | ✅ | No action |

**Bottom line:** The canonical results, A4 correction, and n=10 extension's
headline (DP wins stay significant) are solid. The N1 radius/attack match
correlation (ρ=0.668, p=0.0065) is the one headline finding that does not hold
up to independent recomputation from committed data. This should be fixed or
qualified before submission.
