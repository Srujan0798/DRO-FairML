# Archive policy (HARD RULE)

**Never hard-delete** archive trees. Prefer **move into** archive over `rm`.

Protected paths:
- `docs/_archive/`
- `experiments/_archive/`
- `scripts/_archive/`
- `results/stale_archived/`
- `figures/historical/`
- `paper/_archive/`

Report-live figures (must remain at `figures/` root):
`fig1_main_results`, `fig2_dp_reduction_heatmap`, `fig4_significance_matrix`,
`fig5_accuracy_fairness_tradeoff`, `fig7_summary_win_rates`.

Science: do not rewrite `results/canonical_tau1.json` rows casually.
IF claims are **mixed**; Adult/DP α=0.1 is **5/6**; LSAC/DP degenerate.
