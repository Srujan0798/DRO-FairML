# Wave-1 ablation live progress

_Updated continuously while grids run. Canonical 540 + UTKFace 90 are **locked** and untouched._

## Ship package (ready anytime)
- `paper/main.pdf`, `report/report.pdf`
- `results/canonical_tau1.json` (540)
- `results/utkface_canonical.json` (90 REAL)

## Ablation grids (separate JSON; resume-safe)

| Ablation | File | Target | Status |
|----------|------|--------|--------|
| A4 random vs adversarial | `results/random_vs_adversarial.json` | 144 | RUNNING |
| A3 λ/lr grid | `results/lambda_grid.json` | 72 | RUNNING |
| A5 empirical radii | `results/empirical_radii.json` | 180 | RUNNING |
| N5 K_inner | `results/kinner_ablation.json` | 180 | RUNNING |
| A1 kNN attack_k | `results/knn_ablation.json` | 360 | RUNNING |
| A2 τ ablation | `results/tau_ablation.json` | 360 | RUNNING |

## Summaries (partial OK; re-run when complete)
- `results/*_summary.md` via `experiments/summarize_*.py`
- **Do not** put incomplete multipliers (e.g. 12–40×) in paper until A4 = 144 and n=6 cells complete.

## Rules
1. Never write `canonical_tau1.json` or `utkface_canonical.json` from ablations.
2. One parent process per results file (no dual writers).
3. Negative results are valid — write them honestly.

## Snapshot

- `random_vs_adversarial`: **24/144** (17%)
- `lambda_grid`: **16/72** (22%)
- `empirical_radii`: **49/180** (27%)
- `kinner_ablation`: **16/180** (9%)
- `knn_ablation`: **28/360** (8%)
- `tau_ablation`: **36/360** (10%)

_Last snapshot: 2026-08-04T20:59:16_
