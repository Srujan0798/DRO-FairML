# Pre-Submission Checklist

Run this checklist before every submission.

## Data Integrity
- [ ] `results/all_results.json` has exactly 150 experiments
- [ ] No NaN or Inf values in any metric
- [ ] All individual JSONs in `results/individual/` are valid

## Number Verification (against `results/all_results.json`)
- [ ] LSAC α=0.3: DP reduction ≈ −99.6%, IF reduction ≈ −100%, acc drop ≈ −0.1 pp
- [ ] Credit α=0.3: DP reduction ≈ −91.8%, IF reduction ≈ −96.0%, acc drop ≈ −1.9 pp
- [ ] DP wins (Wilcoxon): 6/9 at α∈{0.1,0.2,0.3}
- [ ] IF wins (Wilcoxon): 5/9 at α∈{0.1,0.2,0.3}
- [ ] α=0.0 shows ties (not wins/losses)
- [ ] Credit α=0.4 DRO accuracy ≥ 0.70

## Report Consistency
- [ ] `report/report.tex` compiles without errors
- [ ] PDF text contains "Credit α=0.3: DP −91.8%, IF −96%" (not −100%)
- [ ] PDF text contains "6/9 DP" and "5/9 IF" (not 7/9 IF)
- [ ] PDF text contains "~37.5×" overhead (not ~54×)
- [ ] Figure captions say "Wilcoxon p<0.05"
- [ ] Adult failure mode is documented (not hidden)

## README Consistency
- [ ] README says "IF in 5/9 cells (Wilcoxon p<0.05)"
- [ ] README notes mean-based 7/9 as parenthetical only
- [ ] README runtime says ~37.5× (not ~54×)

## Submission Package
- [ ] `submission/report.pdf` is IDENTICAL to `report/report.pdf`
- [ ] All 7 figures (fig1–fig7) present in `submission/`
- [ ] `submission/all_results.json` present
- [ ] `submission/table1.csv` present
- [ ] `submission/src/` matches root `src/`
- [ ] `submission/run_experiments.py` present
- [ ] `submission/default.yaml` present
- [ ] 3 prompt documents present in `submission/`

## Validation
```bash
python3 experiments/validate_results.py   # Should print PASS
python3 experiments/verify_theory.py      # Should print ALL VERIFICATIONS PASSED
```

## Git
- [ ] Working tree is clean (`git status` shows nothing)
- [ ] On main branch
- [ ] Latest commit has descriptive message
