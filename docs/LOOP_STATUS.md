# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:10 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** (not clean sweep) · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting brief: `docs/MEETING_2026-08-04.md`
- IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- `main.py`: hard-refuse `--run-experiments` / full-pipeline train unless `FORCE_LEGACY=1`
- `docs/VERIFICATION_REPORT.md`: current banner + historical §0 (no more IF 64/PENDING as live)
- Archive: `paper/sections/_IF_PLACEHOLDER_NOTES.md`, `docs/chat/gchat_raw_export.md` → `docs/_archive/`
- `figures/README.md`: paper-live vs meeting/historical PDFs (no deletes)
- UTKFace REAL progress committed with docs only

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | INDEX complete; archive holds agent noise |
| experiments `*.py` (active) | 16 | Makefile path + runners |
| scripts (active) | 4 | agent_h_finalize, deploy_utkface, extract features, flair2 snippet |
| figures `*.pdf` | ~59 | 5 report-live; rest meeting/appendix — see `figures/README.md` |
| UTKFace REAL rows | **23 / 90** | attack=`dp` only; α=0.3 filling; **no paper claim** |

## Next ticks
1. Optional: archive `run_experiments.py` behind Makefile-only path (still needed for FORCE_LEGACY)
2. Optional: move bulk `logs/*.log` into `logs/archive_root/` (keep recent)
3. UTKFace row count only until multi-attack cells exist
4. Keep pytest + validate green after moves

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
