# Loop status

**Cadence:** every **5 minutes** (job `019fcbd740be`)  
**Updated:** 2026-08-04 ~14:35 IST

## Science (locked)
| Item | State |
|------|--------|
| Grid | **540/540** committed (`4ce1d30`) |
| IF | 180 non-deg (max≈0.239); **mixed** story |
| Adult/DP α=0.1 | **5/6** (still p&lt;0.05) |
| Meeting | `docs/MEETING_2026-08-04.md` |
| Paper/report | IF mixed prose + tables (`7cc3409`) |

## This tick (done)
- Intentional commit: worktree 540 vs old HEAD 383 (+157 IF only; 0 changed cells)
- Archived `results/_if_chunk_*.json` → `results/stale_archived/if_chunks/`
- Archived noise docs → `docs/_archive/cleanup_2026-08-04/`
- Archived finished IF ops scripts + `generate_latex_extras.py`
- Dropped tracked root/doc duplicates already under `_archive/`
- Paper/report/figD regenerated; no “IF pending”
- pytest **62 passed**
- UTKFace REAL local: **18/90** rows (DP α∈{0,0.1,0.2}×6); train live α=0.3; not paper claims; flair2 still SSH

## Next ticks
- Stage 4 optional: unify figure generators / archive unused plot_* once sole Makefile entry clear
- UTKFace progress only until enough REAL rows for any image claim
- Keep tree clean; do not rewrite `canonical_tau1.json`

## Rules
- Do not rewrite `canonical_tau1.json` science
- Archive over delete
- pytest green after moves
- IF is mixed; Adult/DP α=0.1 = 5/6
