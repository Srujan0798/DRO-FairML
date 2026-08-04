# Archive policy (hard rule)

**Never hard-delete** these trees:

- `docs/_archive/`
- `experiments/_archive/`
- `scripts/_archive/`
- `results/stale_archived/`
- `figures/historical/`
- `paper/_archive/`

Prefer **move into** these directories over deleting files.
If a purge is required for size, **git-history must retain** content and a README must
document how to restore (`git checkout <commit> -- <path>`).

**Report-live figures** that must stay at `figures/` root (not only historical/):
`fig1_main_results`, `fig2_dp_reduction_heatmap`, `fig4_significance_matrix`,
`fig5_accuracy_fairness_tradeoff`, `fig7_summary_win_rates`.

Science: never rewrite `results/canonical_tau1.json` rows casually.
