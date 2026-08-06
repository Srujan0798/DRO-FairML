# TASK A — does the AL improvement generalise? (pre-registered)

rows: **42/42** · μ=5 · criterion fixed before data in `docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md`

| dataset | attack | α | n | DP dro | DP AL | R (rel. reduction) | p | acc dro | acc AL | floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | dp | 0.0 | 6 | 0.1426 | 0.0676 | +52.6% | 0.0156* | 0.8147 | 0.8058 | 0.7521 | genuine win |
| adult | dp | 0.3 | 6 | 0.2614 | 0.0731 | +72.0% | 0.0156* | 0.6755 | 0.7321 | 0.7521 | both below floor (excluded regime) |
| adult | dp | 0.4 | 6 | 0.2855 | 0.0551 | +80.7% | 0.0156* | 0.5607 | 0.3495 | 0.7521 | both below floor (excluded regime) |
| adult | if | 0.2 | 6 | 0.0455 | 0.0265 | +41.8% | 0.0156* | 0.7837 | 0.7522 | 0.7521 | **AL DEGENERATE** |
| adult | combined | 0.2 | 6 | 0.1784 | 0.0950 | +46.8% | 0.0156* | 0.7599 | 0.8032 | 0.7521 | genuine win |
| lsac | dp | 0.1 | 6 | 0.2539 | 0.1131 | +55.5% | 0.0156* | 0.9046 | 0.9016 | 0.9016 | both below floor (excluded regime) |
| lsac | dp | 0.2 | 6 | 0.2230 | 0.0659 | +70.5% | 0.0156* | 0.9033 | 0.9016 | 0.9016 | both below floor (excluded regime) |

## Pre-registered rules applied

**Rule 1 — FALSIFICATION (α=0 control).** R(0.0) = **+52.6%** vs pre-registered threshold 20.9% (half of R(0.2)=41.8%). → **Corruption-robustness framing NOT SUPPORTED**: AL helps comparably with no corruption present, so it behaves as a GENERIC fairness regulariser. The robustness framing is withdrawn; AL must be presented as a general fairness improvement, not a corruption-specific one.

**Rule 2 — higher corruption.** α=0.3 p=0.0156, acc 0.7321 (DRO 0.6755); α=0.4 p=0.0156, acc 0.3495 (DRO 0.5607) → **does NOT hold at higher α — the DP gains there are not usable improvements**

> **Instability warning (new failure mode).** At α=0.4 AL's accuracy is 0.3495 against canonical DRO's 0.5607 — a drop of -0.2112. That is far below the constant-predictor floor and below chance for this label balance: μ=5 destabilises training at heavy corruption rather than merely degrading it. Any recommendation for μ must therefore be corruption-dependent (see TASK C), and μ=5 must not be presented as a universal default.

**Rule 3 — other attacks.** IF p=0.0156 (no), COMBINED p=0.0156 (genuine) → **transfers across attacks**

**Rule 4 — LSAC.** Both cells DEGENERATE as predicted — collapse to the constant predictor, consistent with the documented LSAC/DP failure mode. Reported as collapse, NOT as a win.

R = (DP_dro − DP_AL)/DP_dro, the relative DP reduction AL buys. `genuine win` requires p<0.05 AND accuracy clear of the constant-predictor floor by >0.005.
