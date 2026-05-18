# AGENTS.md — DRO-FAIR Project

## Project Overview
This repository implements **DRO-FAIR** (Distributionally Robust Optimization for Fair Classification) from an ICML submission, extended with multi-modal adversarial corruption.

**Key claim:** DRO-FAIR achieves statistically significant DP reductions on Credit (up to −92%) and LSAC (up to −100%) under adversarial corruption, with honest documentation of Adult failure mode.

## Critical Facts for AI Agents

### Authoritative Data Source
- **`results/all_results.json`** — This is THE ground truth. All claims MUST be verified against this file.
- 150 experiments: 3 datasets × 5 alphas × 10 seeds

### Win Criterion
The report uses **Wilcoxon signed-rank test p<0.05** as the official win criterion.
- DP wins: **6/9** cells at α∈{0.1,0.2,0.3}
- IF wins: **5/9** cells at α∈{0.1,0.2,0.3}
- Mean-based wins (for reference only): DP 6/9, IF 7/9

**NEVER claim 7/9 IF while citing Wilcoxon.**

### Headline Numbers (verified from data)
| Dataset | Metric | Value |
|---------|--------|-------|
| LSAC α=0.3 | DP reduction | −99.6% |
| LSAC α=0.3 | IF reduction | −100.0% |
| LSAC α=0.3 | Accuracy drop | −0.1 pp |
| Credit α=0.3 | DP reduction | −91.8% |
| Credit α=0.3 | IF reduction | −96.0% |
| Credit α=0.3 | Accuracy drop | −1.9 pp |

### Known Failure Mode
**Adult α≥0.3:** DRO-FAIR collapses (accuracy ~25-40% on 6/10 seeds) due to adversarial feedback loop. This is NOT a bug — it is an honest, documented empirical limitation. The report explains this in Section 5 (Discussion).

### Hyperparameters
- `lambda_max`: 1.5 (stability fix, was 2.0 in paper)
- `tau_warmup_epochs`: 15 (stability fix, was 10)
- `grad_clip`: 0.5 (stability fix, was 1.0)
- `lambda_decay`: 0.95^epoch (stability fix, not in paper)
- All other params match paper §7.1/§G.4

### Algorithm 1 Implementation Checklist
When modifying `src/training/dro_fair.py`, verify:
1. Step order: θ → λ → p (NOT p → θ → λ)
2. Inner gradient: ∇g only (NOT λ∇g)
3. Bias correction: π_clean = (π_obs − α)/(1 − 2α)
4. Radius: L1-ball radius = 2ρ (NOT ρ)
5. Dykstra: max_iter=500 in tail loop

### Testing
- 32 tests across 4 files
- `python3 -m pytest tests/ -v` to run
- If pytest hangs, try: `python3 -m pytest tests/ -p no:hypothesis`

### Build Commands
```bash
# Run validation
python3 experiments/validate_results.py

# Verify theory
python3 experiments/verify_theory.py

# Generate figures
python3 experiments/generate_figures.py

# Compile report
cd report && tectonic report.tex
```

### Submission Package
The `submission/` folder contains the final deliverables. After any changes:
1. Copy `report/report.pdf` → `submission/report.pdf`
2. Verify all 7 figures are present in `submission/`
3. Verify `submission/src/` matches root `src/`

## File Structure
```
src/training/dro_fair.py       # Algorithm 1
src/training/naive_fair.py     # Baseline
src/corruption/adversarial.py  # PGD + coordinated flips
src/utils/projections.py       # Dykstra projection
src/evaluation/metrics.py      # DP, IF, accuracy
src/data/datasets.py           # Adult, Credit, LSAC loaders
src/models/classifier.py       # MLP
```

## Coding Conventions
- Use `torch.no_grad()` for validation, NOT for training fairness constraints
- Use `tau` MULTIPLY (not divide): `σ(τ·logits)`
- Use local `np.random.RandomState` (not global `np.random.seed`)
- Always clip bias-corrected proportions to [0, 1]
