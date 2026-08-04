# Wave-1 ablation live progress

_Canonical seeds 0–5 (**540**) + UTKFace **90** locked. Loaders default to locked 540._

## Ship package (ready anytime)
- `paper/main.pdf`, `report/report.pdf`
- Claims: Adult/DP α=0.1 **5/6**; IF mixed; LSAC/DP degenerate; UTKFace mixed; α≥0.3 acc Adult/Credit only
- **A3 λ-grid COMPLETE** — findings folded into `paper/sections/appendix_q1_lambda.tex`

## Completed
| Ablation | File | Rows | Finding (honest) |
|----------|------|------|------------------|
| **A3 λ/lr** | `results/lambda_grid.json` | **72/72** | λ_init=0.1 lowers DP and raises acc vs default at α∈{0.2,0.3} (n=6). **No** α=0.3 cell reaches acc>0.7521 (best ~0.685). Keep locked defaults for main protocol; λ_init=0.1 is sensitivity, not a claim change. |

## In flight / partial
| Ablation | File | Target | Snapshot |
|----------|------|--------|----------|
| N2 high-α | `high_alpha_tau.json` | 120 | **LIVE** (orchestrator after A3); ~26 rows |
| A4 RvA | `random_vs_adversarial.json` | 144 | 43 — **≪12×** partial; no paper 12–40× |
| A5 empirical | `empirical_radii.json` | 180 | 69 |
| N5 K_inner | `kinner_ablation.json` | 180 | 24 |
| A1 kNN | `knn_ablation.json` | 360 | 48 |
| A2 τ | `tau_ablation.json` | 360 | 76 |

Orchestrator: `scripts/orchestrate_wave1.sh` → N2 after A3 DONE. Concurrent late parents may still burn CPU; **do not pkill**, **do not launch more**.

## Rules
1. Never write canonical / utkface from ablations.
2. One parent per results file.
3. flair2/NVIDIA parked.

_Last snapshot: 2026-08-04T23:21:00 IST — A3 COMPLETE_
