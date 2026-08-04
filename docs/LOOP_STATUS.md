# Loop status

**Cadence:** every **5 minutes** (job `019fcbd740be`)  
**Updated:** 2026-08-04 ~14:35 IST

## Science (locked)
| Item | State |
|------|--------|
| Grid | **540/540** complete (worktree; intentional commit this tick if not yet on HEAD) |
| IF | 180 non-deg (max≈0.239); **mixed** story |
| Adult/DP α=0.1 | **5/6** (still p&lt;0.05) |
| Meeting | `docs/MEETING_2026-08-04.md` |
| H finalize | done; IF finalize scripts → `scripts/_archive/` |

## This tick
- Verified worktree `canonical_tau1.json`: **540** rows, 180/attack, τ=1, K=10; vs HEAD **383** (only +157 IF rows, **0** changed existing cells)
- Archived `results/_if_chunk_*.json` → `results/stale_archived/if_chunks/` (post-merge leftovers)
- Archived noise docs → `docs/_archive/cleanup_2026-08-04/` (`HARDCODED_NUMBERS_HUNT.md`, `hardcoded_hits.txt`, `AGENT_GUARDRAIL.txt`)
- Archived finished IF ops scripts → `scripts/_archive/` (`agent_h_*`, `finalize_if_sweep.sh`, `watch_sweep_readonly.sh`)
- Archived unused `experiments/generate_latex_extras.py`
- Paper/report already state IF **mixed** (no “IF pending”)
- UTKFace local REAL train: **18/90** rows (DP α∈{0,0.1,0.2}×6 seeds); process live on MPS at DP α=0.3; **not** paper claims yet; flair2 still needs SSH

## Cleanup remaining
- Commit staged chunks: (1) intentional 540 canonical + wilcoxon md + IF chunks archive; (2) paper/report/fig regen; (3) root/docs/scripts archive deletes
- Optional: further merge figure generators (Stage 4); archive more one-off plots once Makefile sole entry is clear
- Keep root free of strays; pytest green after moves

## Rules
- Do not rewrite `canonical_tau1.json` science after 540 commit (additive IF completion only this once)
- Archive over delete
- pytest green after moves
- IF is mixed; Adult/DP α=0.1 = 5/6
