# Clean tree map (post-540 professionalization)

**Updated:** 2026-08-04 · Source of truth: `results/canonical_tau1.json` (540 rows)

## Root (minimal)
```
README.md STATUS.md Makefile LICENSE requirements.txt setup.py main.py
configs/ data/ docs/ experiments/ figures/ logs/ paper/ report/
results/ scripts/ src/ tests/
```

## Where truth lives
| What | Path |
|------|------|
| Full tabular grid | `results/canonical_tau1.json` |
| IF Wilcoxon | `results/if_wilcoxon_summary.txt` |
| Meeting brief | `docs/MEETING_2026-08-04.md` |
| Status | `STATUS.md` |
| Docs map | `docs/INDEX.md` |
| Layout | `docs/REPO_LAYOUT.md` |

## Active code (short)
| Dir | Keep |
|-----|------|
| `src/` | models, trainers, metrics, FairnessTargetedPGD, temperature |
| `experiments/` | run_canonical, run_if_parallel, run_fairness_pgd, loaders, wilcoxon, tables, figures, validate, utkface |
| `scripts/` | deploy_utkface_flair2, extract features, finalize helpers |
| `tests/` | all |

## Archives (not deleted)
- `docs/_archive/` — old prompts, partial IF, agent noise
- `experiments/_archive/` — ~48 one-off runners/plots
- `scripts/_archive/` — watchers/orchestrators
- `results/stale_archived/` — old JSON / if chunks
- `logs/archive_root/` — root logs

## Claims discipline
- IF story is **mixed**, not a three-attack sweep
- Adult/DP α=0.1 is **5/6**
- LSAC/DP is degenerate
- UTKFace only from REAL-tagged rows

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
