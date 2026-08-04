# Correction note for Kuldeep — DRO-FairML

**Date:** 2026-07-20 · **Status:** for the human to review and send. Not sent.
**Source of every number:** `results/canonical_tau1.json` (DP+Combined = 360 complete rows, τ=1.0, k_inner=10, 3 datasets × 5 α × 6 seeds × 2 methods × {DP, Combined}; IF-attack third landing separately — see §1).

You asked on Jun 30 to verify all claims before they went out. That request was well-founded. Three things we reported to you were wrong. Here they are, plainly.

---

## 1. The "individual fairness" (IF) plots sent Jun 30 were mislabelled DP data

On Jun 30 (5:47pm and 5:59pm) we sent `adult_if_*.pdf` as "individual fairness" and quoted *"IF violation: DRO = 0.0195 vs Naive = 0.0177"*.

Those numbers came from the **DP column**, not IF. At that time the IF metric was degenerate — identically ~0 in every row due to a threshold-calibration bug. Under **DP and Combined attacks**, the IF *metric column* is still ~machine zero (max |IF| across those 360 rows is **4.66e-10**). So the project had **no valid IF results** on Jun 30, and the "IF" plots were DP plots under an IF label.

**Current position (2026-08-04):** the IF metric is fixed (cosine-based). A **local parallel IF-attack sweep is in progress** (`experiments/run_if_parallel.py`); partial IF-attack rows are **non-degenerate** (max |if_clean| ≈ 0.098). **No full-grid IF claim is made** until the IF third is complete (180/180) and tables are regenerated. Old Jun-30 IF figures stay withdrawn. This note deliberately quotes **zero IF scientific numbers** as ship claims.

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

- **Adult** (baseline 0.7521), DP attack Naive accuracy: α=0.3 = **0.6669**, α=0.4 = **0.5512** — well below baseline. (DRO at those α is 0.6755 / 0.5607 — still below baseline; do not quote DRO as Naive.)
- **Credit** (baseline 0.7788), DP attack Naive accuracy: α=0.3 = **0.7527**, α=0.4 = **0.7513** — below baseline.

(The baselines 0.7521 / 0.7788 / 0.9016 are the dataset majority-class rates, defined in `experiments/loaders.py` and previously sent to you; the per-α accuracies above are means from `canonical_tau1.json`.)

On **LSAC** the accuracy does **not** fall *below* 0.9016 under the DP attack — it is pinned *at* it (~0.902 at every α), which is the same degeneracy described in §2. The “below constant predictor” statement applies to **Adult and Credit**, not LSAC.

**Defensible claim is scoped to α ≤ 0.2.** Anything led with the α=0.4 gap is scientifically empty.

---

## What IS solid (verified, n=6) — lead with this

- **Adult / DP:** DRO wins every α at p ≤ 0.031 (**α=0.1 is 5/6**, seed 2 loses; **others 6/6**). At the defensible bound (α ≤ 0.2): Naive 0.1491 / DRO 0.1426 (p=0.016) at α=0.0; Naive 0.2026 / DRO 0.1999 (p=0.031) at α=0.1; Naive 0.2452 / DRO 0.2334 (p=0.016) at α=0.2.
- **Adult / Combined:** 6/6 wins at every α, p=0.016.
- **Credit:** under the DP and Combined attacks (complete in the committed grid), DRO beats Naive at **every** cell, all p < 0.05 (Combined α=0.1 is 5/6). The IF-attack third is **landing but incomplete** — see §1 — so **"all three attacks" is not a ship claim**; DP and Combined are verified.

## One caveat to flag

Every number above is traceable to `results/canonical_tau1.json`. **No full IF-attack story ships** until the local sweep finishes 180 IF rows and tables are regenerated — we are not quoting IF metric results as completed findings until then.
