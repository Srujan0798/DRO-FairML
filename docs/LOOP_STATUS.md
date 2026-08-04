# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:31 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting: `docs/MEETING_2026-08-04.md` · IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- **REVERTED hard-delete** commit damage: restored `docs/_archive/`, `experiments/_archive/`,
  `scripts/_archive/`, `results/stale_archived/`, `figures/historical/` from pre-delete commit
  (`9f5b0a0`). Hard rule: **archive over delete**.
- Report-live figures remain at `figures/` root
- UTKFace REAL progress only — no paper claim

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | + restored `_archive/` |
| experiments `*.py` | 17 | live; one-offs in `_archive/` |
| scripts (active) | 4 | finalize, deploy, extract, flair2 snippet |
| UTKFace REAL rows | **41 / 90** | {'dp': 30, 'if': 11}; **no paper claim** |

## Next ticks
1. Do not re-run hard-delete of `_archive/` trees
2. UTKFace multi-attack incomplete — row count only
3. pytest after code moves

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
