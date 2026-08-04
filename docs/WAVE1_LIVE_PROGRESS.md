# Wave-1 ablation live progress

_Updated continuously while grids run. Canonical seeds 0–5 (**540**) + UTKFace **90** locked. Ablations never rewrite those paths. Loaders default to locked 540 (`include_extension=False`)._

## Ship package (ready anytime)
- `paper/main.pdf`, `report/report.pdf`
- Locked claims: Adult/DP α=0.1 **5/6**; IF mixed; LSAC/DP degenerate; UTKFace mixed; α≥0.3 acc Adult/Credit only
- No 12–40× in paper (A4 partial still ≪12×)

## Writers (do not pkill)

| Job | Status |
|-----|--------|
| A3 λ (`run_a3_lambda.py`) | **LIVE** — orchestrator parent + late second parent observed (same JSON). Disk currently **no key dups**. Per-file merge-lock restored in code for *future* processes. |
| A1 kNN, A2 τ, A4 RvA, A5 empirical, N5 K_inner, N2 high-α | **LIVE** concurrent (launched ~23:05; oversubscribes CPU) |
| Original orchestrator `scripts/orchestrate_wave1.sh` | Still running A3 phase |

**Attention:** machine-wide `_AblationLock` was briefly bypassed in `run_ablation_parallel.run`; **restored** this tick. Also added per-JSON re-read+key-merge on append. Running processes still use in-memory code from launch — do not start more parents.

## Ablation grids

| Ablation | File | Target | Snapshot |
|----------|------|--------|----------|
| A3 λ/lr | `lambda_grid.json` | 72 | **65/72** (α=0.2 full; α=0.3 filling) |
| A4 RvA | `random_vs_adversarial.json` | 144 | 43/144 |
| A5 empirical | `empirical_radii.json` | 180 | 69/180 |
| N5 K_inner | `kinner_ablation.json` | 180 | 23/180 |
| A1 kNN | `knn_ablation.json` | 360 | 48/360 |
| A2 τ | `tau_ablation.json` | 360 | 76/360 |

## Partial findings (honest)
- **A3 α=0.2 (n=6 complete):** λ_init=0.1 beats default on DP+acc; no α=0.3 acc>0.7521 yet (max≈0.70).
- **A4:** multipliers 0.2–1.1× on partial Adult — **not** 12–40×.
- **A5 / N5 / A1 / A2:** incomplete; no new main-claim text.

## Rules
1. Never write canonical / utkface from ablations.
2. One parent per results file — do not launch seconds.
3. Never pkill `run_a*` / orchestrator / JSON writers.
4. flair2/NVIDIA parked.

_Last snapshot: 2026-08-04T23:08:00 IST_
