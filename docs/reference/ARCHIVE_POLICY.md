# Archive policy (final — 2026-08-04)

This repo previously oscillated for ~8 commits between "hard-delete archives" and
"restore archives" as different agent sessions disagreed. That war caused a real
regression: `experiments/generate_figures.py` — the only script that regenerates the
5 figures embedded in `report.tex` — was swept into `experiments/_archive/` as a
"one-off," then lost when a later purge hard-deleted `_archive/` without checking
what was inside. It has been restored (2026-08-04) to `experiments/` (live, not
archived).

**The rule, once, for good:**

1. **Git history is the archive.** This repo does not maintain a living
   `_archive/`/`stale_archived/`/`historical/` directory in the working tree.
   Superseded files are deleted outright — recoverable any time via
   `git log --oneline --all -- <path>` then `git show <commit>:<path>`.
2. **Before deleting or moving any `experiments/*.py`, `scripts/*`, or
   `paper/`/`report/` file, grep for it first**: check the `Makefile`, `main.py`,
   other scripts, and every `.tex` file for a reference or an `\includegraphics`/
   `\input` call whose target it produces. If it's the only generator for something
   currently embedded in a paper/report artifact, it stays live — even if it looks
   like a one-off. This is exactly the check that would have caught the
   `generate_figures.py` deletion.
3. **Report-live figures** that must always exist at `figures/` root (never deleted
   without regenerating first): `fig1_main_results`, `fig2_dp_reduction_heatmap`,
   `fig4_significance_matrix`, `fig5_accuracy_fairness_tradeoff`,
   `fig7_summary_win_rates`. Their generator is `experiments/generate_figures.py`.
4. **Science files are sacred, always**: never rewrite `results/canonical_tau1.json`
   or `results/utkface_canonical.json` rows casually, and never touch either while
   an experiment process is actively appending to it (`ps aux | grep run_`).
