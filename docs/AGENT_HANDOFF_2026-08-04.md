# Agent handoff — 2026-08-04 (post Aug 10 final gate)

**Aligned with:** Grok session + Claude Code session `00b1a526…`  
**HEAD:** `origin/main` — keep polishing toward Aug 10 (science frozen).

## Meeting (done)
Present from: `docs/MEETING_HANDOUT_2026-08-04.md`  
Figures: `figures/fig_tau1_headline.pdf`, `figures/fig_final_wilcoxon_table.pdf`.

Honest lines: Adult/DP **α=0.1 = 5/6**; **IF metric** Adult/Credit incl. **α=0.3** (6/6); DP-under-IF **mixed**; LSAC/DP **degenerate**; UTKFace **90/90 REAL mixed pilot**.

---

## DONE (do not re-open)
| Item | Status |
|------|--------|
| Tabular **540** (dp/if/combined = 180) | ✅ frozen — **never write** `canonical_tau1.json` |
| Real IF numbers | ✅ IF metric formalized + DP-under-IF mixed (`if_wilcoxon_summary.txt`) |
| **Agent V** mismatches + Jul-2 figs | ✅ STATUS / KULDEEP / VERIFICATION / figs fixed & pushed |
| Paper assembly | ✅ figures + auto tables + appendices in `paper/main.pdf` |
| Seed-paired Wilcoxon fix | ✅ `generate_report_tables.build_wilcoxon` pairs by seed |
| flair2 access / L40S / code / features | ✅ proven — **PARKED** (no torch install; don’t restart) |
| **Local UTKFace REAL** | ✅ **90/90** — `results/utkface_canonical.json` + `utkface_summary.md` |
| Paper / report Aug 10 narrative | ✅ mixed UTKFace pilot; τ=1 / 5/6 / IF@0.3 / LSAC |
| Final gate | ✅ `make test` (64) + `validate` PASS + paper + report |

---

## Aug 10 (submission package)
| Share | Path |
|-------|------|
| Paper | `paper/main.pdf` |
| Report | `report/report.pdf` |
| Handout | `docs/MEETING_HANDOUT_2026-08-04.md` |
| Optional image summary | `results/utkface_summary.md` |

Checklist: `docs/AUG10_SUBMISSION_CHECKLIST.md`  
Advisor map: `docs/ADVISOR_CONCERNS_CHECKLIST.md`  
Live board: `STATUS.md`

---

## Optional later (not blocking Aug 10)
- flair2 pixel-level / end-to-end only if greenlit
- Empirical-radii full table (Q5 appendix exists; canonical stays uniform)

---

## Don’t
- Don’t retrain or rewrite `canonical_tau1.json`
- Don’t wait on / restart flair2 torch download
- Don’t report synthetic UTKFace as real
- Don’t claim “6/6 every α” or clean three-attack mirror
- Don’t claim UTKFace is Adult-style low-α DP sweep (it’s **mixed**)
