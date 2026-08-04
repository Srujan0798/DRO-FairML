# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:05 UTC (~14:35 IST)

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** (not clean sweep) · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting brief: `docs/MEETING_2026-08-04.md`

## This wave
- Paper/report IF + UTKFace prose scrub (honest mixed IF; REAL partial UTKFace only)
- Archived tau_ablation-dependent plotters → `experiments/_archive/`
- Archived `scripts/watch_sweep_readonly.sh` (IF sweep done)
- Staged: fig generators already → `_archive/`; `finalize_experiments.py` → scripts archive
- `docs/INDEX.md` + `CLEAN_TREE.md` kept current

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | + `_archive/`, `chat/` |
| experiments `*.py` (active) | 16 | + `_archive/` one-offs |
| scripts (active, non-archive) | 4 | agent_h_finalize, deploy_utkface, extract features, flair2 snippet |
| figures `*.pdf` | ~59 | live; `stale_archived/` empty |
| UTKFace REAL rows | **21 / 90** | attack=dp only; α≤0.3 partial; **no paper claim** |

## Next ticks
1. Commit paper/report scrub + archive moves when PDFs rebuild clean
2. Optional: thin `main.py` deprecation for `--run-experiments`
3. UTKFace continue REAL grid only; report row count only
4. Keep `make validate` + `pytest -q` green after any code move

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
