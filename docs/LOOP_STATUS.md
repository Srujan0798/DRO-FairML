# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:30 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting: `docs/MEETING_2026-08-04.md` · IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- **Blocked mass delete** of `docs/_archive/`, `experiments/_archive/`, `scripts/_archive/`, `results/stale_archived/` — restored from HEAD (hard rule: archive over delete)
- Restored logs FINALIZATION + agent_h inventory, `paper/_archive/ICML_submission.pdf`, `results/reductions.json`
- Report-live figs still present at `figures/` root
- UTKFace REAL progress only (no paper claim)

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | INDEX complete |
| experiments `*.py` | 17 | + `plot_meeting_figs_540.py` |
| scripts (active) | 4 | finalize, deploy, extract, flair2 snippet |
| UTKFace REAL rows | **41 / 90** | dp=30, if=11; **no paper claim** |

## Next ticks
1. Do **not** hard-delete `_archive/` trees — git-history only is not “archive over delete”
2. Optional: document `plot_meeting_figs_540.py` in experiments/README
3. Keep report-live PDFs at figures/ root
4. pytest after code moves

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
