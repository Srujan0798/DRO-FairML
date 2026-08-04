# Figures (LIVE)

## Regenerated from 540-row canonical (2026-08-04) — prefer these for meeting
- `fig_tau1_headline.pdf` — Adult DP, wins [6,5,6,6,6]
- `fig_final_wilcoxon_table.pdf` / `figD10_final_wilcoxon_table.pdf` — DP win matrix
- `figD1`–`figD4` — constant-predictor / tradeoff (deliverables regen)
- `fig1`, `fig2`, `fig4`, `fig5`, `fig7`, `main_results`, `test_time_eval` — report includes (regen 2026-08-04)

## Older stamps (optional / incomplete data)
- `fig_final_constant_predictor_*`, `fig_final_lambda_*`, `fig_final_tradeoff_*` — mtime **Jul 2** (pre-540); prefer `figD*` where overlapping
- `figD5`–`figD7` convergence — **Jul 20**; need `results/individual/` histories (not present) — do not claim as 540-fresh
- `fig_acc_win_curves_tau1.pdf` — **Jul 2**
- `fig_high_alpha_*` — optional high-α panels

Regenerate meeting pair:
```bash
python3 experiments/plot_meeting_figs_540.py
```
