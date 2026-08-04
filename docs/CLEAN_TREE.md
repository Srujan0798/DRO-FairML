# Clean tree map (post-540 professionalization)

**Updated:** 2026-08-04 · Source of truth: `results/canonical_tau1.json` (**540** rows)

## Root (minimal)
```
README.md  STATUS.md  Makefile  LICENSE  requirements.txt  setup.py  main.py
configs/  data/  docs/  experiments/  figures/  logs/
paper/  report/  results/  scripts/  src/  tests/
```

## Where truth lives
| What | Path |
|------|------|
| Full tabular grid | `results/canonical_tau1.json` |
| Wilcoxon (DP/IF/Combined) | `results/canonical_wilcoxon.{csv,md}`, `results/if_wilcoxon_summary.txt` |
| Derived tables | `results/table1_*`, `results/summary_stats.csv` (regen via `make tables` / `make results`) |
| UTKFace (REAL only) | `results/utkface_canonical.json` |
| Meeting brief | `docs/MEETING_2026-08-04.md` |
| Status board | `STATUS.md` |
| Docs index | `docs/INDEX.md` |
| Layout (one screen) | `docs/REPO_LAYOUT.md` |

## Active code
| Dir | Keep (role) |
|-----|-------------|
| `src/` | Trainers, `FairnessTargetedPGD`, metrics, models, radii, `temperature.py` (τ=1) |
| `experiments/` | Canonical pipeline + Makefile artifact gens + UTKFace + headline plots (see below) |
| `scripts/` | **5** operational helpers only (see below) |
| `tests/` | Full pytest suite (`make test`) |
| `main.py` | Legacy CLI: `make results` → `generate_results.py` |

### `experiments/` live set (~16 `.py`)
| Group | Files |
|-------|-------|
| Runners | `run_canonical.py`, `run_if_parallel.py`, `run_fairness_pgd.py`, `run_canonical_empirical.py`, `run_utkface.py`, `run_utkface_server.py`, `run_experiments.py` (legacy / `FORCE_LEGACY=1`) |
| I/O + stats | `loaders.py`, `canonical_to_all_results.py`, `compute_canonical_wilcoxon.py`, `validate_results.py`, `meeting_summary.py`, `verify_theory.py` |
| Makefile artifacts | `generate_results.py` (`make results`), `generate_report_tables.py` (`make tables`), `generate_all_deliverables.py` (`make deliverables`) |
| Archived plots | Headline `plot_*` + `generate_final_figures` / `generate_figures` / `generate_fig8_matrix` live under `experiments/_archive/` |

### `scripts/` live set (5)
| File | Role |
|------|------|
| `agent_h_finalize.sh` | Post-540 finalize / gate |
| `watch_sweep_readonly.sh` | Read-only IF/count poller |
| `deploy_utkface_flair2.sh` | flair2 UTKFace deploy |
| `extract_utkface_features.py` | ResNet feature extract |
| `flair2_ssh_config_snippet.txt` | SSH Host snippet |

## `results/` live (claims only)
```
canonical_tau1.json          ★ 540-row grid
canonical_wilcoxon.{csv,md}
if_wilcoxon_summary.txt
all_results.json             nested bridge for legacy plot path
table1_*.tex/csv  summary_stats.csv
utkface_canonical.json
stale_archived/              old knn/lambda/if_chunks/if_poc/partials — do not claim
```

## Archives (not deleted)
| Path | Contents |
|------|----------|
| `docs/_archive/` | Agent prompts, MASTER plans, old handoffs, chat strays |
| `experiments/_archive/` | One-off runners, old fig gens (`generate_figures`, `generate_fig8_matrix`, latex extras, …) |
| `scripts/_archive/` | Watchers, orchestrators, `finalize_experiments.py`, cluster IF re-run |
| `results/stale_archived/` | Partial JSON, knn/lambda ablations, IF chunks, POC rows |
| `logs/` (+ `archive_root/`) | Run logs |

## Makefile map → scripts
| Target | Entry |
|--------|--------|
| `test` | `pytest tests/` |
| `validate` | `experiments/validate_results.py` |
| `wilcoxon` | `experiments/compute_canonical_wilcoxon.py` |
| `tables` | `experiments/generate_report_tables.py` |
| `results` | `main.py --generate-results` → `generate_results.py` |
| `deliverables` | `experiments/generate_all_deliverables.py` |
| `paper` / `report` | tectonic |
| `full` | wilcoxon + tables + results + deliverables (**no retrain**) |

## Claims discipline
- IF story is **mixed**, not a clean three-attack sweep
- Adult/DP α=0.1 is **5/6** (still p&lt;0.05)
- LSAC/DP is degenerate (majority collapse)
- UTKFace claims only from **REAL**-tagged rows
- Never cite `results/stale_archived/` or τ=100 ablations as main claims

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
