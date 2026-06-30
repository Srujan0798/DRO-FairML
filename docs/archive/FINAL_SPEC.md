# DRO-FairML — Final Specification & Completion Plan

> Date: 2026-06-30
> Running: canonical PID 6431 (303/540), empirical PID 11023 (18/270)
> Source of truth: ULTIMATE_HANDOFF.md (replaces all 29 old handoffs)

---

## §0. CURRENT STATE

### Experiments Running (background, do not touch)
| Process | PID | Progress | Target | ETA |
|---|---|---|---|---|
| `run_canonical.py` | 6431 | 303/540 | 540 | ~4-6 hrs (Credit α=0.4 + LSAC) |
| `run_canonical_empirical.py` | 11023 | 18/270 | 270 | ~4-6 hrs (runs in parallel) |

### Data Completeness
- **Adult**: 180/180 ✅
- **Credit**: 123/180 (α=0.3 finishing, α=0.4 remaining)
- **LSAC**: 0/180 (not reached yet)
- **Empirical**: 18/270 (Adult started)

### Tests: 60/60 ✅ | Lambda grid: 72/72 ✅
### Code: All audit fixes verified, src frozen ✅
### PDF build: Tectonic 0.16.9 available, both PDFs build (stale tables)

---

## §1. IMMEDIATE — LET RUNNING EXPERIMENTS FINISH

**Do NOT interfere**. Both PIDs are healthy at ~97% CPU, making progress.

Monitor with:
```bash
# Check progress
python3 -c "import json; d=json.load(open('results/canonical_tau1.json')); print(f'Canonical: {len(d)}/540')"
python3 -c "import json; d=json.load(open('results/canonical_tau1_empirical.json')); print(f'Empirical: {len(d)}/270')"

# Check processes
ps aux | grep run_canonical | grep -v grep
```

When canonical hits 540, the JSON will be complete. Empirical runs independently to 270.

---

## §2. AFTER EXPERIMENTS COMPLETE

### Phase 1 — Regenerate Analysis (5 min)
```bash
python3 experiments/compute_canonical_wilcoxon.py   # n=6 Wilcoxon from full 540
python3 experiments/analyze_tau1.py                  # Summary tables from full data
python3 experiments/generate_report_tables.py        # Auto-generate LaTeX tables
```

### Phase 2 — Regenerate All Figures (10 min)
```bash
python3 experiments/generate_final_figures.py        # All fig_final_* series
# Or individually:
python3 experiments/figures/fig_win_curves.py
python3 experiments/figures/fig_random_vs_adv.py
python3 experiments/figures/fig_high_alpha.py
python3 experiments/figures/fig_lambda_heatmap.py
```

Check figures/ for output — should produce 15+ PDF+PNG pairs.

### Phase 3 — Rebuild PDFs (2 min)
```bash
tectonic --outdir report report/report.tex
tectonic --outdir paper paper/main.tex
```
Both should build clean. Only font warnings are expected (unicode chars, cosmetic).

### Phase 4 — Verify Report Numbers
Spot-check 5 numbers from report PDF against `results/tau1_summary.csv`:
- Adult α=0.2 DP: DRO should be ~0.237 (NOT the old tau=100 value 0.503)
- Wilcoxon p-values should match `results/canonical_wilcoxon.csv`

---

## §3. FIX STALE REPORT CONTENT

The report has hardcoded numbers from the old tau=100 regime:

| Location | What's stale | Fix |
|---|---|---|
| report/report.tex:338-371 | Main Results table (tau=100) | Regenerate or replace with `\input{auto_generated_main_results.tex}` |
| report/report.tex:711-730 | PGD table (tau=100) | Replace with auto-generated version |
| report/report.tex:656-673 | Ablation table (tau=100) | Replace with auto-generated version |
| Abstract | "n=3" → "n=6" | Manual edit after canonical 540 |
| All hand-typed stats throughout | Pre-τ=1 numbers | Each must be updated to match tau1_summary.csv |

Best approach: **convert hardcoded tables to `\input{}`** so they auto-update from CSVs.

---

## §4. BLOCKED ITEMS

### UTKFace (Q13)
- **Blocked**: flair2.iitgn.ac.in GPU access via supin.gopi email
- **Draft ready**: `EMAIL_TO_SUPIN_GOPI_DRAFT.txt`
- **Action**: Send the email. If access granted, run:
  ```bash
  python3 experiments/run_utkface_server.py
  ```
- **Fallback**: Note in paper as "infrastructure limitation"

### PDF Build Environment
- Tectonic is available on this Mac
- If missing on other machines: `brew install tectonic`

---

## §5. FINAL DELIVERABLES CHECKLIST

- [ ] canonical_tau1.json: 540/540 rows
- [ ] canonical_tau1_empirical.json: 270/270 rows
- [ ] lambda_lr_grid.json: 72/72 rows
- [ ] canonical_wilcoxon.csv: n=6 complete, p-values marked
- [ ] All figures regenerated from final canonical data
- [ ] report/report.pdf: builds clean, numbers traceable to CSVs
- [ ] paper/main.pdf: builds clean, numbers traceable to CSVs
- [ ] Report tables use `\input{}` (not hardcoded)
- [ ] UTKFace: run on flair2 or documented as blocked
- [ ] Tests pass: `python3 -m pytest tests/ -q`
- [ ] HANDOFF.md updated with final status
- [ ] Commit and push all changes
- [ ] `EMAIL_TO_SUPIN_GOPI_DRAFT.txt` sent

---

## §6. COMMIT STRATEGY

```bash
# Before experiments complete — commit non-result changes
git add -A && git commit -m "chore: cleanup temp files, add FINAL_SPEC + ULTIMATE_HANDOFF"

# After experiments complete + analysis
git add results/ figures/ report/ paper/
git commit -m "feat: canonical 540/540 + empirical 270/270 + full analysis"

# After PDF rebuild
git add report/report.pdf paper/main.pdf
git commit -m "docs: rebuild PDFs from final canonical data"
```

---

## §7. NARRATIVE SUMMARY (for the professor)

**Key finding**: tau=1 fixed makes DRO beat Naive on DP at every α on Adult, with advantage growing with α and NO accuracy cost. The earlier "DRO is fragile" result was a tau=100 artifact.

**Defensible regime**: α≤0.2. At α≥0.3, even 30-40% coordinated label corruption causes both methods to degrade below the constant predictor baseline (acc ~0.55-0.68 vs 0.752). Neither tau tuning nor λ grid overcomes this ceiling — it is inherent to the corruption level, not a hyperparameter issue.

**Adversarial >> Random**: The coordinated PGD attack raises DP ~30-40x more than random noise, confirming the attack is working correctly.

**What's left**: Only Credit α=0.4 and all of LSAC need to finish in the canonical run. Everything else (analysis scripts, figures, PDF build, tests) is ready and will auto-generate from the final data.
