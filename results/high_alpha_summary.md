# Agent N2 — high-α rescue (Kuldeep Jun-16 3-step protocol)

Analysis-only. No new training. Source: `results/high_alpha_tau.json` (26/120 rows). Adult constant-predictor acc = **0.7521**.

Kuldeep, Jun 16 (verbatim): "Different tau value 1st if not improving then change learning rates for lamda or something else check loss convergence plots and choose according to it on validation set".

## Coverage

- STEP 1 (τ ∈ {2,5,20}): 26/72 rows
- STEP 2 (lr_lambda=0.01): 0/24 rows
- STEP 3 (epochs=200, dump_history): 0/24 rows
- **INCOMPLETE** — partial-data mode; re-run as rows land (idempotent).

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
| 0.3 | 2.0 | dro | 3 | 0.6779 | 0.3936 | ✗ |
| 0.3 | 5.0 | dro | 4 | 0.6759 | 0.5174 | ✗ |
| 0.3 | 20.0 | dro | 4 | 0.6772 | 0.5531 | ✗ |
| 0.3 | 2.0 | naive | 5 | 0.6682 | 0.3996 | — |
| 0.3 | 5.0 | naive | 5 | 0.6661 | 0.4938 | — |
| 0.3 | 20.0 | naive | 5 | 0.6635 | 0.5262 | — |

**Verdict (STEP 1):** 
INCOMPLETE (26/72 rows) — cannot conclude yet.

## STEP 2 — does lr_lambda=0.01 help at high α?

lr_lambda=0.01 (canonical 5e-3), Adult, dp, α∈{0.3,0.4}, 6 seeds, 2 methods. Compare to canonical rows above (τ=1.0, lr=5e-3).

| α | method | n | acc | dp | Δacc vs canonical | Δdp vs canonical |
|---|---|---|---|---|---|---|

**Verdict (STEP 2):** 
INCOMPLETE (0/24 rows) — cannot conclude yet.

## STEP 3 — convergence diagnostics (epochs=200, dump_history)

History JSONs present: DRO=0, naive=0 (only DRO history is dumped by run_fairness_pgd.py dump_history code; naive is in-process only).

Test-set cells at epochs=200:

| α | method | n | acc | dp | acc>0.7521? |
|---|---|---|---|---|---|

**Q3(i): does val_loss plateau before epoch 60 (underfitting) or keep decreasing (would benefit from more epochs)?**

| α | method | n_hist | verdict | slope_early | slope_late | val_loss@1 | val_loss@60 | val_loss@end |
|---|---|---|---|---|---|---|---|---|

**Verdict Q3(i):** INCOMPLETE — no DRO α=0.3 history JSONs yet.

**Q3(ii): does val_acc at epoch 200 exceed 0.7521?**

| α | method | n | mean val_acc@200 | best val_acc | acc>0.7521? |
|---|---|---|---|---|---|

**Verdict Q3(ii):** INCOMPLETE — no val_acc@200 data yet.

## Headline

INCOMPLETE (0/24 STEP 3 rows) — headline will be set once STEP 3 lands.
