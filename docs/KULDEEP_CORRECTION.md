# Correction note for Kuldeep — DRO-FairML

**Date:** 2026-07-20 · **Status:** for the human to review and send. Not sent.
**Source of every number:** `results/canonical_tau1.json` (360 rows, τ=1.0, k_inner=10, 3 datasets × 5 α × 6 seeds × 2 methods × {DP, Combined} attacks).

You asked on Jun 30 to verify all claims before they went out. That request was well-founded. Three things we reported to you were wrong. Here they are, plainly.

---

## 1. The "individual fairness" (IF) plots sent Jun 30 were mislabelled DP data

On Jun 30 (5:47pm and 5:59pm) we sent `adult_if_*.pdf` as "individual fairness" and quoted *"IF violation: DRO = 0.0195 vs Naive = 0.0177"*.

Those numbers came from the **DP column**, not IF. At that time the IF metric was degenerate — identically ~0 in every row due to a threshold-calibration bug. In the committed data, the maximum |IF violation| across all 360 rows is **4.66e-10** (i.e. zero to machine precision). So the project had **no valid IF results** on Jun 30, and the "IF" plots were DP plots under an IF label.

**Current position:** the IF metric is now fixed (cosine-based) in code, but the real IF experiment has **not yet been run** — it is waiting on a cluster compute job (Agent G1). Until that run produces the first real IF numbers, **no IF claim is made and the old IF figures stay withdrawn.** This note deliberately contains zero IF numbers.

## 2. LSAC was reported "pending" but is in fact complete — and it is a loss (and degenerate)

LSAC under the **DP attack** is finished and the result is negative:

| α | Naive DP | DRO DP | DRO wins? | p (1-sided) |
|---|---|---|---|---|
| 0.0 | 0.1447 | 0.1829 | 0/6 | 1.000 |
| 0.1 | 0.2201 | 0.2539 | 0/6 | 1.000 |
| 0.2 | 0.1827 | 0.2230 | 0/6 | 1.000 |
| 0.3 | 0.1827 | 0.2220 | 0/6 | 1.000 |
| 0.4 | 0.1827 | 0.2211 | 0/6 | 1.000 |

DRO loses at **every** α, 0/6 seeds, p=1.0. Two further facts make this worse than a plain loss:
- The **Naive DP** value is **bit-identical (0.1827)** at α = 0.2, 0.3, 0.4 — the metric stops responding as corruption triples, which means the model has collapsed, not that it is robust.
- **Accuracy is pinned to the constant-predictor baseline (~0.9016)** at every α on LSAC. The model *is* the majority-class predictor. So the LSAC/DP "result" measures a degenerate classifier, not a fairness comparison. (Diagnosis: `docs/LSAC_DEGENERACY.md`.)

This is **not** hidden now: LSAC/DP is reported as a degenerate/diagnostic result, not as a DRO win or loss.

By contrast, **LSAC/Combined is a genuine win** — DRO lower DP than Naive, p=0.0156 at α=0.1/0.3/0.4 (and p=0.031 at α=0.2). That is the honest LSAC result to lead with.

## 3. The α ≥ 0.3 regime is below the constant-predictor baseline — the "advantage grows with α" framing is empty there

At α ≥ 0.3 both DRO and Naive drop **below** the constant-predictor (majority-class) accuracy on Adult and Credit, so any "DRO is better" claim in that regime is comparing two useless models. From the committed canonical:

- **Adult** (baseline 0.7521): Naive accuracy at α=0.3 ≈ 0.676, α=0.4 ≈ 0.608 — well below baseline.
- **Credit** (baseline 0.7788): Naive accuracy at α=0.3 ≈ 0.757, α=0.4 ≈ 0.744 — below baseline.

(The baselines 0.7521 / 0.7788 / 0.9016 are the dataset majority-class rates, defined in `experiments/loaders.py` and previously sent to you; the per-α accuracy drops above are read directly from `canonical_tau1.json`.)

On **LSAC** the accuracy does not fall *below* 0.9016 — it is pinned *at* it (~0.902 at every α under the DP attack), which is the same degeneracy described in §2.

**Defensible claim is scoped to α ≤ 0.2.** Anything led with the α=0.4 gap is scientifically empty.

---

## What IS solid (verified, n=6) — lead with this

- **Adult / DP:** DRO wins at every α. At the defensible bound (α ≤ 0.2): Naive 0.1491 / DRO 0.1426 (p=0.016) at α=0.0; Naive 0.2026 / DRO 0.1999 (p=0.031) at α=0.1; Naive 0.2452 / DRO 0.2334 (p=0.016) at α=0.2. All six α are 6/6 wins, p ≤ 0.031.
- **Adult / Combined:** 6/6 wins at every α, p=0.016.
- **Credit:** under the DP and Combined attacks (the two attacks present in the committed grid), DRO beats Naive at **every** cell, all p < 0.05. (The IF-attack column is still pending the cluster re-run — see §1 — so "all three attacks" is not yet fully in hand; DP and Combined are verified.)

## One caveat to flag

Every number above is traceable to `results/canonical_tau1.json`. The **IF-attack results are withdrawn** until Agent G1's cluster re-run produces the first real ones — we are not quoting any IF figure, corrected or otherwise, until then.
