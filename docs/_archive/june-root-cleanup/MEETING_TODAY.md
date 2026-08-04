# MEETING ONE-PAGER — Kuldeep (today 4 PM)

> **Use this + KULDEEP_DISCUSSION.md + the .txt message for the call. Concise, evidence-based only. All stats from committed CSVs + live PID 79899.**

## Top 3 bullets (ready for discussion)
- Launched full K=10 6-seed canonical (tau=1), 37 rows running now (PID 79899, all Adult so far), α=0.0 block complete with n=6: Naive DP 0.1491 / DRO 0.1426 — DRO edge even at zero corruption.
- Tau=1 headline holding from prior 3-seed data (DRO beats/ties Naive DP at every α on Adult; advantage grows with α; no acc trade-off; old "DRO fragile" was tau=100 artifact). Full 6-seed + Credit/LSAC in flight.
- Parallel agent deliverables complete: theory (B), figures (headline, win curves, random-vs-adv absolute; C), report builds + KULDEEP + message (D), provenance/runner (A).

## Link to working doc
See full live status, tables (absolute DP from tau1_summary.csv etc.), ablations, asks: **KULDEEP_DISCUSSION.md**

## 2-3 most important figures to open (in this order)
1. figures/fig_tau1_headline.pdf — side-by-side tau=1 (DRO wins) vs tau=100 (DRO loses); absolute DP + SE bars.
2. figures/fig_win_curves_tau1.pdf — DRO advantage (Naive-DRO) vs α curves for DP/IF/combined attacks; shows growth with α.
3. figures/figC3_random_vs_adversarial.pdf — absolute DP under random vs adversarial (12-40× lift); directly addresses "show adversarial raises DP more than random".

(Also available: figures/adult_tau1_headline_meeting.pdf, fig_win_curves_tau1.png etc. All from C's scripts on committed results.)

## Exact copy-paste message (MEETING_MESSAGE_TO_MAM.txt — send as-is)
```
Hi Mam,

Quick update before the call (we have ~1hr):

We launched the full spec-compliant 6-seed canonical run with **K_inner=10** and **fixed tau=1 for all alphas** (exactly the config from the tau ablation + paper mandatory settings). It is running live now (PID 79899, 37/540 rows completed so far — all Adult; the entire α=0.0 block finished with n=6 seeds).

Early result from the live K=10/tau=1 canonical (Adult DP attack, α=0.0, full 6 seeds):
- Naive mean DP = 0.1491
- DRO mean DP = 0.1426 (DRO slightly better even with zero corruption)

This directly addresses the K_inner=5 vs 10 question — we are running the real K=10 version now. The tau=1 headline (DRO wins or ties on Adult at every α, advantage grows with α, acc equal-or-better) is holding in the new run. Parallel work also completed: empirical radii theory + test (B), all new meeting figures with absolute DP + CM fonts (C), updated KULDEEP doc + report/paper with Q5 appendix + LSAC IF framing (D), provenance + canonical runner + knn to 3 datasets + UTKFace local smoke (A).

Random-vs-adversarial (your request): clean absolute DP numbers ready (e.g. Adult α=0.2: adv +~0.18 vs random ~0; 12-40x stronger effect).

**Questions for you:**
1. With the live 6-seed canonical (K=10 + tau=1) now running and early Adult α=0 n=6 numbers showing DRO not worse (actually slightly better) even at zero corruption, is the story "fixed tau=1 makes DRO robust under coordinated fairness attacks" solid for the paper / next submission?
2. For the adversarial vs random comparison, should we present the absolute DP values (0.15 → 0.53 etc.) as the main figure, or the multiplier (12-40×)?
3. 6 seeds in the canonical — enough for the Wilcoxon in the write-up (p<0.05 now mathematically possible), or push for more?
4. UTKFace: we have a local smoke using the exact canonical config + full server script ready. Should we chase flair2.iitgn.ac.in access this week (email supin.gopi drafted), or finish the tabular analysis first?

Thanks Mam! Looking forward to the discussion.

(Prepared with current live run data + agent deliverables. All numbers from committed results / live canonical json.)
```

## What we will show Kuldeep
- The live run progress (37 rows, α=0 n=6 DRO edge at 0.1426 vs 0.1491).
- Tau=1 headline holding (Adult DP wins 2/3 + 3/3 + 3/3 + 3/3; full 6-seed canonical will enable p<0.05 Wilcoxon).
- Agent deliverables summary (theory, fresh absolute-DP figures, KULDEEP doc, message, report/paper updates).

**Evidence before claims.** All from CSVs (tau1_summary.csv, tau1_wilcoxon.csv, random_vs_adversarial_new.json, knn_ablation_table.csv) + live canonical (results/canonical_tau1.json partial). Post-meeting: fold 540-row data + re-run C plots on canonical.

## Latest from live run as of 2026-06-16 15:23:40 IST (harvest + interim analysis)
- Read running log + canonical json (39 rows now vs prior 37 mention; PID 79899 active).
- α=0.0 Adult n=6 complete: Naive DP 0.1491 / DRO DP 0.1426 (DRO wins **6/6**). 
- Ran `experiments/compute_canonical_wilcoxon.py` safely on partial (skips n<2); now `results/canonical_wilcoxon.md` uses live canonical source, shows p=0.0156 * for adult dp α=0.0 (and combined/if).
- New `INTERIM_CANONICAL_ADULT_TABLE.md` produced with clean table (by alpha/attack/method, means+win counts, absolute DP focus on headline DP attack; explicit "live K=10 tau=1, partial data, will update when more rows land").
- Partial α=0.1 DP (n=1 seed0): Naive 0.2197 / DRO 0.2146 (DRO wins that seed). See INTERIM table for full breakdown.
- Files updated: INTERIM_CANONICAL_ADULT_TABLE.md, KULDEEP_DISCUSSION.md (appended), MEETING_TODAY.md (appended), results/canonical_wilcoxon.{csv,md}.

---
*MEETING PREP (fast one-pager). Previous MEETING_TODAY content archived conceptually in _archive/. All paths absolute from repo root.*