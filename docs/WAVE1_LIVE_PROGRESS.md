# Wave-1 ablation live progress

_Updated continuously while grids run. Canonical seeds 0–5 (540) + UTKFace 90 are **locked** and untouched by ablations. File may show >540 if N10 seed-extension rows were appended earlier; ablations never rewrite those paths._

## Ship package (ready anytime)
- `paper/main.pdf`, `report/report.pdf`
- `results/canonical_tau1.json` — locked science = seeds 0–5 (**540**); file currently **560** (+20 partial N10 extension, seeds 6–9, adult/DP partial)
- `results/utkface_canonical.json` (90 REAL)

## Ablation grids (separate JSON; resume-safe)

| Ablation | File | Target | Status |
|----------|------|--------|--------|
| A3 λ/lr grid | `results/lambda_grid.json` | 72 | **LIVE** (orchestrator workers=12) |
| N2 high-α | `results/high_alpha_tau.json` | — | queued after A3 |
| A5 empirical radii | `results/empirical_radii.json` | 180 | queued (partial on disk) |
| L2 LSAC radii | `results/lsac_radii_fix.json` | — | queued |
| A4 random vs adversarial | `results/random_vs_adversarial.json` | 144 | queued (partial on disk) |
| A2 τ ablation | `results/tau_ablation.json` | 360 | queued (partial on disk) |
| N5 K_inner | `results/kinner_ablation.json` | 180 | queued (partial on disk) |
| A1 kNN attack_k | `results/knn_ablation.json` | 360 | queued (partial on disk) |
| N1 attack strength | `results/attack_strength.json` | — | queued |
| S N10 extension | append-only to canonical | +360 tabular | queued last |

Orchestrator: `scripts/orchestrate_wave1.sh` (sequential, `ABLATION_WORKERS=12`). One parent per JSON.

## Summaries (partial OK; re-run when complete)
- `results/*_summary.md` via `experiments/summarize_*.py` (now importable without manual `PYTHONPATH` thanks to `experiments/__init__.py` + `sys.path` inserts)
- **Do not** put incomplete multipliers (e.g. 12–40×) in paper until A4 = 144 and n=6 cells complete.
- Partial A4 multipliers (Adult only, incomplete): **≪12×** — paper already uses qualitative pilot language only.

## Honest partial findings (not ship claims until complete)
- **A3 λ**: α=0.2 cells suggest λ_init=0.1 can improve DP+acc vs default; α=0.3 rescue of acc>0.7521 still **unanswered** (few α=0.3 rows).
- **A5 empirical radii**: empirical vs uniform coordinated **no DP gain** on available cells (pilot; incomplete).
- **A1 kNN**: larger attack_k raises IF violation (partial Adult cells); DP not raised.
- **N5 K_inner**: available Adult cells show **no material DP change** K∈{5,10,20} (partial).
- **A2 τ**: τ=1 Wilcoxon from canonical confirms Adult/DP **5/6 at α=0.1**; τ=10/100 partial Adult only.
- **A4 RvA**: 12–40× **not supported** on partial Adult α∈{0.1,0.2} cells — do not re-introduce into paper.

## Rules
1. Never write `canonical_tau1.json` or `utkface_canonical.json` from ablations.
2. One parent process per results file (no dual writers).
3. Negative results are valid — write them honestly.
4. Never pkill `run_a*`, orchestrator, or JSON writers mid-flight.

## Snapshot

- `random_vs_adversarial`: **43/144** (30%)
- `lambda_grid`: **43/72** (60%; **LIVE writer** — A3 in orchestrator)
- `empirical_radii`: **69/180** (38%)
- `kinner_ablation`: **23/180** (13%)
- `knn_ablation`: **48/360** (13%)
- `tau_ablation`: **76/360** (21%)

_Last snapshot: 2026-08-04T22:55:00 IST_
