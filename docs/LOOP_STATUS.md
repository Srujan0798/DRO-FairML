# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:15 UTC

## Locked science
- canonical **540/540** committed — never rewrite science rows
- IF **mixed** (not clean sweep) · Adult/DP α=0.1 is **5/6** · LSAC/DP degenerate
- Meeting brief: `docs/MEETING_2026-08-04.md`
- IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## This wave
- **STATUS.md** UTKFace: local MPS REAL partial grid (not “probe only / flair2 blocked”)
- **UTKFACE_STATUS:** **26/90** REAL (`dp` α≤0.3 complete; α=0.4 filling)
- **logs hygiene:** 144 old batch/chunk logs → `logs/archive_root/bulk_pre_540/` (kept recent utkface/IF/H)
- Paper/report: only legitimate pending (UTKFace full grid; Q5 empirical JSON)

## Counts
| Area | Count | Notes |
|------|------:|-------|
| Root files (non-dir) | 7 | LICENSE Makefile README STATUS main.py requirements setup |
| docs live `*.md` | 21 | INDEX lists live set |
| experiments `*.py` (active) | 16 | Makefile + runners |
| scripts (active) | 4 | finalize, deploy, extract, flair2 snippet |
| figures `*.pdf` | ~59 | 5 report-live; see `figures/README.md` |
| logs top-level | ~14 | + `archive_root/` bulk |
| UTKFace REAL rows | **27 / 90** | attack=`dp` only; **no paper claim** |

## Next ticks
1. Optional: keep or thin `meeting_summary.py` (CLI utility, not Makefile)
2. UTKFace: when `dp` 30/30, still no paper claim until if/combined exist
3. pytest + validate if code moves
4. Do not push science rewrites of `canonical_tau1.json`

## Reproduce (no retrain)
```bash
make install && make data && make test && make validate && make paper && make report
```
