# DRO-FairML

**Distributionally Robust Optimization for Fairness** under adversarial group/attribute corruption.

## What this is

Two training methods, compared under the same attack:

| Method | Role |
|--------|------|
| **DRO-FAIR** | Min-max Lagrangian DRO with corruption-calibrated total-variation (TV) uncertainty sets (Algorithm 1). |
| **Naive-FAIR** | Fairness-regularized baseline *without* DRO robustness. |

**Attack:** `FairnessTargetedPGD` — adversarial fairness-targeted projected gradient descent on the training labels / attributes (modes: **DP**, **IF**, **Combined**). Random corruption is *not* the evaluation method.

**Datasets:** Adult, Credit, LSAC (tabular). UTKFace is a **real image-feature pilot** (90/90 REAL ResNet18 features; mixed clean-test; not an Adult copy-paste claim).

**Metrics:** demographic parity (DP), individual fairness (IF), accuracy.

---

## Canonical configuration (locked)

| Parameter | Value |
|-----------|-------|
| τ (temperature) | **1.0** |
| K_inner | **10** |
| epochs | 60 |
| PGD steps | 20 |
| seeds | **n = 6** |
| λ init | 0.0 |
| radii_mode | uniform |
| coordinated | False |

**Complete grid (committed):**  
`3 datasets × 5 α × {DP, IF, Combined} × 6 seeds × 2 methods` = **540 rows** in  
[`results/canonical_tau1.json`](results/canonical_tau1.json).

You do **not** need to retrain the 540-row grid to use the paper or report. Results and derived artifacts are in the repo.

---

## Results (honest summary)

Source of truth: `results/canonical_tau1.json` + Wilcoxon in `results/canonical_wilcoxon.*` / `results/if_wilcoxon_summary.txt`. Full meeting write-up: [`docs/MEETING_2026-08-04.md`](docs/MEETING_2026-08-04.md). Live board: [`STATUS.md`](STATUS.md).

| Claim region | Finding |
|--------------|---------|
| **Adult & Credit, α ≤ 0.2** | DRO improves DP vs Naive under **DP** and **Combined** attacks (paired Wilcoxon, n=6; note Adult/DP α=0.1 is **5/6**, still p&lt;0.05). |
| **LSAC / DP** | **Degenerate** — DRO collapses toward majority predictor; not a method win. See [`docs/LSAC_DEGENERACY.md`](docs/LSAC_DEGENERACY.md). |
| **IF attack** | **MIXED (split metrics)** — cosine IF non-degenerate (max \|if_clean\| ≈ 0.24). **IF metric:** Adult/Credit win at α∈{0.1–0.4} incl. **α=0.3** (6/6, p=0.0156). **DP under IF:** Adult wins α≤0.2 but **loses α=0.3**; LSAC loses α≤0.3. Not a clean three-attack DP sweep. See `results/if_wilcoxon_summary.txt`. |
| **α ≥ 0.3** | Both methods can fall below the constant-predictor accuracy baseline on Adult/Credit → no strong method claim in that regime. |
| **UTKFace** | **REAL 90/90** (`results/utkface_canonical.json`); clean-test DP **mixed** (significant DRO wins mainly at high α). See `results/utkface_summary.md`. |

Earlier “DRO is fragile” plots used stepped **τ=100** (temperature artifact). Canonical claims use **τ=1**.

---

## How to reproduce

Preferred path: install → data → tests → validate → rebuild paper/report from **committed** results.

```bash
# 1. Environment (Python ≥ 3.10)
make install          # pip install -r requirements.txt
# PDF builds need tectonic:  brew install tectonic

# 2. Tabular data (Adult, Credit, LSAC)
make data             # download + SHA-256 verify

# 3. Unit tests
make test

# 4. Consistency / Wilcoxon checks on committed JSON
make validate
make wilcoxon         # rewrite results/canonical_wilcoxon.{csv,md}

# 5. Regenerate tables, figures, PDFs (does NOT retrain)
make tables
make results
make deliverables
make paper            # paper/main.pdf
make report           # report/report.pdf

# One-shot artifact regen (no training):
make full
```

### Optional: retrain (not required)

Full recompute is multi-hour on CPU and is only for audit / extension:

```bash
python3 experiments/run_canonical.py          # τ=1, K=10, 6 seeds → canonical_tau1.json
python3 experiments/run_canonical.py --smoke  # tiny smoke run, not for claims
```

### Progress check

```bash
make monitor
# expect: total=540, attacks dp/if/combined = 180 each
```

---

## Repository layout

| Path | Contents |
|------|----------|
| [`src/`](src/) | Core library: DRO-FAIR / Naive-FAIR trainers, `FairnessTargetedPGD`, metrics, radii. |
| [`experiments/`](experiments/) | Canonical runner, ablations, plots, validation, deliverable generators. |
| [`results/`](results/) | Committed experiment JSON (incl. `canonical_tau1.json`), Wilcoxon tables. |
| [`figures/`](figures/) | Paper / meeting figures (PDF). |
| [`paper/`](paper/) | ICML-style paper (`main.tex` → `main.pdf`). |
| [`report/`](report/) | Longer report (`report.tex` → `report.pdf`). |
| [`docs/`](docs/) | Meeting brief, verification, design notes (`docs/reference/` for planning). |
| [`tests/`](tests/) | Unit / e2e tests. |
| [`data/`](data/) | Download script + raw tabular inputs. |
| [`configs/`](configs/) | Default YAML. |
| [`scripts/`](scripts/) | Orchestration / server helpers (not needed for default repro). |
| [`logs/`](logs/) | Runtime logs only (`logs/README.md`; gitignored). |

One-screen map: [`docs/reference/REPO_LAYOUT.md`](docs/reference/REPO_LAYOUT.md).

### Key entry points

- `src/training/dro_fair.py` — DRO-FAIR (Algorithm 1)
- `src/training/naive_fair.py` — Naive-FAIR baseline
- `src/corruption/adversarial.py` — `FairnessTargetedPGD`
- `experiments/run_canonical.py` — writes `results/canonical_tau1.json`
- `experiments/validate_results.py` / `compute_canonical_wilcoxon.py` — checks & stats

---

## Status & meeting brief

- **[`STATUS.md`](STATUS.md)** — single source of truth for completion state and remaining work.
- **[`docs/MEETING_2026-08-04.md`](docs/MEETING_2026-08-04.md)** — meeting brief with verified tables (honest 5/6 cells, IF MIXED).
- **[`docs/INDEX.md`](docs/INDEX.md)** — doc index.
- **[`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md)** — claim → data audit.

---

## Constraints (do not violate)

- Evaluation corruption is **adversarial** (`FairnessTargetedPGD`), not random as the method.
- No oracle leak: DRO sees only the corruption budget α (and known attack structure for radii) — never the true per-sample mask.
- Canonical training defaults: epochs=60, K_inner=10, step order θ→λ→p, λ init 0.0.
- Private academic repo; no publicity without PI approval.

---

## License

See [`LICENSE`](LICENSE).
