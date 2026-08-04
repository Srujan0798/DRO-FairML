# Agent N2 — high-α rescue (Kuldeep Jun-16 3-step protocol)

Analysis-only. No new training. Source: `results/high_alpha_tau.json` (120/120 rows). Adult constant-predictor acc = **0.7521**.

Kuldeep, Jun 16 (verbatim): "Different tau value 1st if not improving then change learning rates for lamda or something else check loss convergence plots and choose according to it on validation set".

## Coverage

- STEP 1 (τ ∈ {2,5,20}): 72/72 rows
- STEP 2 (lr_lambda=0.01): 24/24 rows
- STEP 3 (epochs=200, dump_history): 24/24 rows

## STEP 1 — per-α τ: does any τ lift α=0.3 DRO acc above 0.7521?

Canonical τ=1.0 (read-only, from `results/canonical_tau1.json`):

| α | method | n | acc | dp |
|---|---|---|---|---|
| 0.3 | dro | 6 | 0.6755 | 0.2614 |
| 0.3 | naive | 6 | 0.6669 | 0.2848 |
| 0.4 | dro | 6 | 0.5607 | 0.2855 |
| 0.4 | naive | 6 | 0.5512 | 0.3140 |

STEP 1 cells (τ ∈ {2,5,20}, Adult, dp, α∈{0.3,0.4}, 6 seeds, 2 methods). ✓ = DRO acc > 0.7521 (Adult constant predictor).

| α | τ | method | n | acc | dp | acc>0.7521? |
|---|---|---|---|---|---|---|
| 0.3 | 2.0 | dro | 6 | 0.6752 | 0.3863 | ✗ |
| 0.3 | 5.0 | dro | 6 | 0.6750 | 0.5077 | ✗ |
| 0.3 | 20.0 | dro | 6 | 0.6742 | 0.5540 | ✗ |
| 0.4 | 2.0 | dro | 6 | 0.5586 | 0.4241 | ✗ |
| 0.4 | 5.0 | dro | 6 | 0.5554 | 0.5114 | ✗ |
| 0.4 | 20.0 | dro | 6 | 0.5539 | 0.5247 | ✗ |
| 0.3 | 2.0 | naive | 6 | 0.6665 | 0.3972 | — |
| 0.3 | 5.0 | naive | 6 | 0.6647 | 0.4898 | — |
| 0.3 | 20.0 | naive | 6 | 0.6624 | 0.5225 | — |
| 0.4 | 2.0 | naive | 6 | 0.5495 | 0.4382 | — |
| 0.4 | 5.0 | naive | 6 | 0.5458 | 0.5076 | — |
| 0.4 | 20.0 | naive | 6 | 0.5437 | 0.5166 | — |

**Verdict (STEP 1):** 
**No.** No τ ∈ {2,5,20} lifts α=0.3 DRO accuracy above 0.7521. Per-α τ tuning alone does not rescue high-α — proceed to STEP 2.

## STEP 2 — does lr_lambda=0.01 help at high α?

lr_lambda=0.01 (canonical 5e-3), Adult, dp, α∈{0.3,0.4}, 6 seeds, 2 methods. Compare to canonical rows above (τ=1.0, lr=5e-3).

| α | method | n | acc | dp | Δacc vs canonical | Δdp vs canonical |
|---|---|---|---|---|---|---|
| 0.3 | dro | 6 | 0.6770 | 0.2577 | +0.0015 | -0.0036 |
| 0.3 | naive | 6 | 0.6664 | 0.2804 | -0.0005 | -0.0044 |
| 0.4 | dro | 6 | 0.5580 | 0.2799 | -0.0027 | -0.0056 |
| 0.4 | naive | 6 | 0.5501 | 0.3114 | -0.0010 | -0.0026 |

**Verdict (STEP 2):** 
lr_lambda=0.01 **raises** DRO α=0.3 acc to 0.6770 (canonical 0.6755); but still below 0.7521 (constant predictor).

## STEP 3 — convergence diagnostics (epochs=200, dump_history)

History JSONs present: DRO=12, naive=0 (only DRO history is dumped by run_fairness_pgd.py dump_history code; naive is in-process only).

Test-set cells at epochs=200:

| α | method | n | acc | dp | acc>0.7521? |
|---|---|---|---|---|---|
| 0.3 | dro | 6 | 0.7065 | 0.2765 | ✗ |
| 0.3 | naive | 6 | 0.6994 | 0.2903 | ✗ |
| 0.4 | dro | 6 | 0.6246 | 0.3099 | ✗ |
| 0.4 | naive | 6 | 0.6094 | 0.3329 | ✗ |

**Q3(i): does val_loss plateau before epoch 60 (underfitting) or keep decreasing (would benefit from more epochs)?**

| α | method | n_hist | verdict | slope_early | slope_late | val_loss@1 | val_loss@60 | val_loss@end |
|---|---|---|---|---|---|---|---|---|
| 0.3 | dro | 6 | still_decreasing (6/6) | -0.000771 | -0.000245 | 0.6856 | 0.6111 | 0.5729 |
| 0.4 | dro | 6 | still_decreasing (6/6) | 0.000564 | -0.000310 | 0.6993 | 0.7394 | 0.6923 |

**Verdict Q3(i):** DRO α=0.3 val_loss is **still decreasing** past epoch 60 (6/6 seeds) — 60 fixed epochs UNDERFITS at high corruption; the model would benefit from more epochs.

**Q3(ii): does val_acc at epoch 200 exceed 0.7521?**

| α | method | n | mean val_acc@200 | best val_acc | acc>0.7521? |
|---|---|---|---|---|---|
| 0.3 | dro | 6 | 0.7056 | 0.7210 | ✗ |
| 0.4 | dro | 6 | 0.6239 | 0.6321 | ✗ |

**Verdict Q3(ii):** **No** — DRO α=0.3 val_acc@200 = 0.7056 ≤ 0.7521 (n=6). Even at epoch 200 the model does not clear the constant-predictor baseline.

## Headline

**STEP 3 does NOT lift DRO α=0.3 accuracy above 0.7521** (acc=0.7065, n=6). The α≥0.3 limitation now has **convergence evidence** (epochs=200, val-monitored) instead of an assertion — Kuldeep's requested check is closed with data.

## Convergence plots (literal artifact Kuldeep requested)

- `results/high_alpha_convergence_a0.3.png`
- `results/high_alpha_convergence_a0.4.png`
