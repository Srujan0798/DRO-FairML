# Wave-1 ablation live progress

_Updated continuously while grids run. Canonical seeds 0–5 (**540**) + UTKFace **90** are **locked**. On-disk `canonical_tau1.json` may be 560 (+20 partial N10); loaders default to locked 540. Ablations never rewrite those paths._

## Ship package (ready anytime)
- `paper/main.pdf`, `report/report.pdf`
- Locked science: seeds 0–5 in `results/canonical_tau1.json` (**540**); `results/utkface_canonical.json` (**90 REAL**)
- Claims: Adult/DP α=0.1 **5/6**; IF mixed; LSAC/DP degenerate; UTKFace mixed; α≥0.3 acc claim Adult/Credit only

## Ablation grids (separate JSON; resume-safe)

| Ablation | File | Target | Status |
|----------|------|--------|--------|
| A3 λ/lr grid | `results/lambda_grid.json` | 72 | **LIVE** — α=0.2 complete (36/36); α=0.3 filling |
| N2 high-α | `results/high_alpha_tau.json` | — | queued after A3 |
| A5 empirical radii | `results/empirical_radii.json` | 180 | queued (69 on disk) |
| L2 LSAC radii | `results/lsac_radii_fix.json` | — | queued |
| A4 random vs adversarial | `results/random_vs_adversarial.json` | 144 | queued (43 on disk) |
| A2 τ ablation | `results/tau_ablation.json` | 360 | queued (76 on disk) |
| N5 K_inner | `results/kinner_ablation.json` | 180 | queued (23 on disk) |
| A1 kNN attack_k | `results/knn_ablation.json` | 360 | queued (48 on disk) |
| N1 attack strength | `results/attack_strength.json` | — | queued |
| S N10 extension | append-only canonical | +360 tabular | queued last |

Orchestrator: `scripts/orchestrate_wave1.sh` (sequential, `ABLATION_WORKERS=12`). One parent per JSON.

## Partial findings (honest; not full-grid claims)
- **A3 λ (α=0.2 complete, n=6):** λ_init=0.1 improves DP and acc vs default (λ_init=0, lr=0.005); λ_init=0.01/lr=0.005 mild DP gain. **α=0.3:** no acc>0.7521 yet (max≈0.70); incomplete.
- **A4 RvA:** partial Adult multipliers **≪12×** — paper keeps qualitative pilot language only.
- **A5 empirical radii:** no DP gain vs uniform on available cells (pilot).
- **A1 kNN:** larger k raises IF violation (partial Adult).
- **N5 K_inner / A2 τ:** incomplete; τ=1 locked story unchanged (5/6 at Adult/DP α=0.1).

## Rules
1. Never write `canonical_tau1.json` or `utkface_canonical.json` from ablations.
2. One parent process per results file (no dual writers).
3. Negative results are valid — write them honestly.
4. Never pkill `run_a*`, orchestrator, or JSON writers mid-flight.
5. Do not put incomplete multipliers (e.g. 12–40×) in paper until A4=144 + n=6 cells.

## Snapshot

- `random_vs_adversarial`: **43/144** (30%)
- `lambda_grid`: **54/72** (75%; **LIVE** — α=0.2 done; α=0.3 filling)
- `empirical_radii`: **69/180** (38%)
- `kinner_ablation`: **23/180** (13%)
- `knn_ablation`: **48/360** (13%)
- `tau_ablation`: **76/360** (21%)

_Last snapshot: 2026-08-04T22:56:00 IST_
