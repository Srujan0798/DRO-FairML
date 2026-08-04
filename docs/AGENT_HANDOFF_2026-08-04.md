# Agent handoff — 2026-08-04 (post 15:14)

## Meeting (16:00)
**Present only from:** `docs/MEETING_2026-08-04.md`  
Also share: `figures/fig_tau1_headline.pdf`, `figures/fig_final_wilcoxon_table.pdf`.

Honest lines: Adult/DP **α=0.1 = 5/6**; IF **mixed**; LSAC/DP **degenerate**.

---

## DONE (do not re-open)
| Item | Status |
|------|--------|
| Tabular **540** (dp/if/combined = 180) | ✅ frozen — **never write** `canonical_tau1.json` |
| Real IF numbers | ✅ mixed story in MEETING + `if_wilcoxon_summary.txt` |
| **Agent V** mismatches + Jul-2 figs | ✅ STATUS / KULDEEP / VERIFICATION / figs fixed & pushed |
| Repo cleanup | ✅ slim tree; archives purged; `origin/main` updated |
| flair2 access / L40S / code / features | ✅ proven — **PARKED** (no torch install; don’t restart) |

---

## IN PROGRESS
| Item | Notes |
|------|--------|
| **Local UTKFace REAL** | One resume runner only. DP done; IF then Combined → **90** rows. See `docs/UTKFACE_STATUS.md`. |

When 90/90: commit `results/utkface_canonical.json`, honest summary, optional paper blurb — **never synthetic**.

---

## Aug 10 (after UTKFace decision)
1. Paper/report polish (K)  
2. Final claim trace (L)  
3. UTKFace in paper **or** honest scope-out  
4. flair2 only if GPU-heavy experiment greenlit  

---

## Don’t
- Don’t retrain or rewrite `canonical_tau1.json`  
- Don’t wait on / restart flair2 torch download  
- Don’t report synthetic UTKFace as real  
- Don’t run multiple UTKFace writers (race on JSON)  
