# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:25 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting: `docs/MEETING_2026-08-04.md` · IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- **Critical fix:** report-live figures (`fig1/2/4/5/7`) restored from `figures/historical/` after clutter purge moved them
- `make results` OK from canonical (90 nested DP rows → tables/plots)
- `figures/README.md` corrected (report-live vs historical)
- if_chunks intermediate shards documented (merged into 540; git history retains)
- UTKFace REAL progress (row count only)

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | INDEX complete |
| experiments `*.py` | 16 | + README |
| scripts (active) | 4 | finalize, deploy, extract, flair2 snippet |
| figures root PDFs | ~30 | + `historical/` |
| UTKFace REAL rows | **38 / 90** | dp=30, if=8; **no paper claim** |

## Next ticks
1. Keep report-live figs at `figures/` root (never only under historical/)
2. `make deliverables` figD5–D9 fail-loud without optional inputs — OK
3. UTKFace: no claim until multi-attack cells reviewed
4. pytest after code moves

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
