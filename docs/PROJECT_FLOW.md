# DRO-FairML — Clear Project Flow

## Overview (from full chat history)
DRO vs Naive under adversarial fairness-targeted PGD (DP/IF/combined attacks).
Core insight (per Kuldeep): **fixed tau=1** (not stepped high-tau) makes DRO robust for α ≤ 0.2 on Adult (DP wins/advantage grows, acc ≥ 0.78 stable). At α ≥ 0.3: inherent 30-40% corruption ceiling (acc drops below constant predictor ~0.752 even with different tau or lambda tuning). Different tau first for high-α to test improvement, then lambda/lr or convergence plots.

All runs: K_inner=10, full provenance on every row, adversarial only (no Random as method), absolute metrics.

## 1. Experiment Launch Flow (CPU-heavy, one-writer)
- Main canonical (6 seeds, fixed τ=1): `python experiments/run_canonical.py --k_inner 10`
  - Produces: results/canonical_tau1.json (full rows with dataset/alpha/seed/attack/method/acc/dp/...)
- Lambda grid (Q1 hyperparams): `python experiments/run_lambda_lr_grid.py`
  - Produces: results/lambda_lr_grid.json + history_*.json
- High-α tau ablations (Kuldeep priority for high-α acc vs constant): `python experiments/run_tau_ablation.py --tau 5 --alphas 0.3 0.4 ...`
  - tau_ablation_tau*.json
- Other: run_knn_ablation.py, run_fairness_pgd.py (legacy), UTKFace server scripts.

Active runners monitored via PIDs (16334=lambda, 21531=canonical). Never pkill; use orchestration.

Live logs in root (lambda_lr_grid.log, canonical.log) + detailed in logs/.

## 2. Data & Provenance
- All results/*.json have full row provenance (k_inner, tau, radii_mode, lambda_init, etc.).
- Summaries: results/tau1_summary.csv, tau1_wilcoxon.csv, knn_*, high_alpha_*
- Stale/individual/history/ in results/ kept for audit (never delete during runs).

## 3. Analysis & Tables
- `python experiments/analyze_tau1.py` — processes canonical/ablations → summaries + wilcoxon on completed blocks.
- `python experiments/generate_report_tables.py` — auto .tex for report/paper.
- `python experiments/compute_canonical_wilcoxon.py`
- Partial safe on completed α (e.g. adult 0.0/0.1 full n=6).

## 4. Visualization (meeting/Kuldeep format)
- `python experiments/plot_tau1_headline.py`, `plot_high_alpha_tau.py`, `plot_win_curves_tau1.py`, etc.
- `python experiments/generate_final_figures.py` (Claude Agent C script) — constant-predictor, tradeoff, lambda heatmaps (0.3/0.4), wilcoxon, etc. (CM serif, SE bars, x=α, absolute values, no grids).
- Outputs: figures/ (adult_*_meeting.pdf/png, fig_high_alpha_*, fig_final_*, etc.)
- Comfort dups only in FRIEND/ + kuldeep_meeting/ (not source of truth).

## 5. Reporting & Paper
- `cd report && tectonic report.tex` (or via generate_pdf_report.py)
- `cd paper && tectonic main.tex`
- Auto sections from generate_*_tables.py
- Key narrative docs: KULDEEP_DISCUSSION.md (tables + asks), HANDOFF.md (full state + constraints)

## 6. Automation & Final Polish (when thresholds met)
- Orchestrators/watchers (no early action):
  - lambda_watcher.py → finalize + commit at 72/72
  - canonical_watcher.py + advancer → launch empirical on first Credit/LSAC
  - grok_final_delivery_orchestrator.py (in logs/) → polls for 72 + 540 + empirical → runs:
    - finalize_experiments.py
    - generate_final_figures.py + analyze + tables
    - update HANDOFF.md (exact FINAL DELIVERY section from history)
    - targeted git add (figures/, results/*summary*, docs/, report/, paper/ — **not** live JSONs)
    - commit + FINAL_DELIVERY_EVIDENCE.txt
- Scripts in scripts/: finalize_experiments.py, finish_everything_when_ready.sh
- Monitors in logs/ + background subs (Grok + prior).

## 7. Full Run / Monitor Commands
```bash
# Status
python3 finalize_experiments.py status
tail -f logs/grok_final_delivery_orchestrator.log

# When ready (auto by orchestrator)
python3 finalize_experiments.py
python3 experiments/generate_final_figures.py
# then commit as per script
```

## 8. Project Structure (after cleanup)
Root (minimal):
- README.md, HANDOFF.md, KULDEEP_DISCUSSION.md, MASTER_PLAN.md, SERVER_RUNBOOK.md, EMAIL_TO_*, LICENSE, requirements.txt, setup.py, Makefile, main.py
- scripts/ (orchestration: finalize, finish, watchers)
- experiments/ (runners + plotters + _archive)
- results/, figures/, data/, configs/
- docs/ (design notes + _archive/ for all history + project_management/ for status MDs + PROJECT_FLOW.md + CHAT_HISTORY...)
- paper/, report/, submission/, tests/, src/
- logs/ (all .log + watcher .py + .pid)
- FRIEND/, kuldeep_meeting/ (laptop comfort dups only — never commit full source)

No root clutter. After any work: archive transients to docs/_archive/, keep root scannable.

## Current Snapshot (as of last orchestrator poll)
lambda 53/72, canonical 85/540 (adult α0.2 advancing). Low-α adult complete n=6. High-α ceiling locked. All automation waiting correctly.

See KULDEEP_DISCUSSION.md for tables + exact Kuldeep asks alignment.
See docs/project_management/ for live orchestrator notes.

## Latest (POLL#5 23:37)
lambda=54/72 (α0.3 now 18/18 complete!)
canonical=85/540 (adult α0.2=13 rows)
All automation healthy, waiting correctly per design.

## POLL#6 @ 2026-06-17 23:42
lambda=54/72 (α0.3 complete)
canonical=85/540 (adult α0.2 progressing slowly)
Orchestrator still correctly waiting, no Credit/LSAC. Structure cleaned, flow doc up to date.
