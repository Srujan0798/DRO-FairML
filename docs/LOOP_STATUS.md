# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:20 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting: `docs/MEETING_2026-08-04.md` · IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- `configs/default.yaml`: `coordinated: false` (matches all 540 canonical rows)
- `experiments/README.md`: Makefile critical path + runners map
- Paper checklist + meeting brief: post-540 / UTKFace partial REAL (not flair2-probe-only)
- UTKFace REAL progress committed (row count only)

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | INDEX complete |
| experiments `*.py` | 16 | + README.md |
| scripts (active) | 4 | finalize, deploy, extract, flair2 snippet |
| figures `*.pdf` | ~59 | 5 report-live — `figures/README.md` |
| UTKFace REAL rows | **29 / 90** | attack=`dp` only; α=0.4 nearly full; **no paper claim** |

## Next ticks
1. When `dp` hits 30/30, still no claim until if/combined
2. Optional: wire or drop unused `paper/auto_generated/*.tex` inputs
3. Keep pytest green after code moves

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
