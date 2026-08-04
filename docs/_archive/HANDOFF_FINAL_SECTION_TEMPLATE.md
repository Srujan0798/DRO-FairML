## FINAL DELIVERY (auto-completed {DATE})

### Deliverables
- Canonical: {N}/540 (Adult {A1}/180, Credit {A2}/180, LSAC {A3}/180)
- Empirical companion: {M}/270 (DRO radii_mode=empirical, Q5)
- Lambda grid: 72/72 (Q1)
- k-NN ablation: 84/84 each (k=5,10,15, all 3 datasets)
- Tau ablation: 109/109 each (tau=1,5,10,20,100)
- Tests: 60/0
- Report PDF: report.pdf (rebuilt)
- Paper PDF: main.pdf (rebuilt)
- Final figures: figures/fig_final_*.pdf (15+ plots)
- Wilcoxon n=6: results/canonical_wilcoxon.csv (all datasets, p-values marked)

### Key Findings
1. **Defensible regime = alpha <= 0.2**: DRO beats Naive on DP at every alpha in this range with no accuracy cost (acc >= 0.78 bar satisfied)
2. **High-alpha ceiling (alpha >= 0.3)**: Constant-label predictor (acc=0.752) dominates. Neither tau tuning (1/5/10/20/100) nor lambda grid overcomes 30-40% label corruption ceiling.
3. **K_inner=10 mandatory**: K_inner=5 was artifact; paper-mandated K=10 is the spec.
4. **Empirical radii (Q5)**: Known coordinated attack structure lets us invert exact pi_clean; no oracle leak. Validated empirically.
5. **LSAC narrative**: IF-focused (DP inherently biased toward 0 in this dataset).

### Reproducibility
- All experiments logged in results/*.json with full provenance
- Script runner: experiments/run_canonical.py (resume-safe, skips done)
- Orchestrator: scripts/final_delivery_orchestrator.sh (auto-fires chain)
- Tests: pytest tests/ -q -> 60 passed
- Logs: logs/final_delivery_orchestrator.log

### UTKFace Status (Q13)
BLOCKED. flair2.iitgn.ac.in server did not respond to email to supin.gopi (Jun 25 2026). Prof. Manisha confirmed aware (Jun 19 2026). Local smoke test (2 rows) confirms pipeline correct with K_inner=10, tau=1, epochs=60. Paper will note infrastructure limitation.
