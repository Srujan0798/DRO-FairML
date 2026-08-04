# figures/

## Paper / report (live includes)
Referenced by `report/report.tex` (and rebuilt via `make results` / `make deliverables`):
- `fig1_main_results.pdf`
- `fig2_dp_reduction_heatmap.pdf`
- `fig4_significance_matrix.pdf`
- `fig5_accuracy_fairness_tradeoff.pdf`
- `fig7_summary_win_rates.pdf`

## Meeting / appendix / historical (keep on disk)
- `figD*.pdf`, `figC*.pdf`, `*_meeting.pdf`, `fig_tau1_headline.pdf`, heatmaps, etc.
- Sourced from older tau ablation / lambda grids now in `results/stale_archived/`.
- **Do not delete** until paper appendix stop citing them; regenerate only from
  `results/canonical_tau1.json` when updating claim figures.

## Layout
- Live PDFs/PNGs stay here for TeX `\includegraphics`.
- `stale_archived/` reserved for truly orphaned renames (prefer archive over delete).
