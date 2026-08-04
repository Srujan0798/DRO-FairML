# figures/

## Report-live (must stay at `figures/` root)
Referenced by `report/report.tex`:
- `fig1_main_results.pdf`
- `fig2_dp_reduction_heatmap.pdf`
- `fig4_significance_matrix.pdf`
- `fig5_accuracy_fairness_tradeoff.pdf`
- `fig7_summary_win_rates.pdf`

Also used by `make results`: `main_results.pdf`, `test_time_eval.pdf`.

## Deliverables / meeting (canonical-derived)
- `figD1`–`figD4`, `figD10` — regenerate via `make deliverables` (needs canonical)
- `figD5`–`figD9` need live `results/individual/` or lambda grid (not in committed set); existing PDFs kept when present

## Historical (not for primary claims)
- `historical/` — τ=100, meeting one-offs, figC ablations, old dashboards
- Prefer archive over delete. Do not cite as canonical science.

```bash
make results        # tables + main plots from canonical_tau1.json
make deliverables   # figD pack (some tasks fail-loud without optional inputs)
```
