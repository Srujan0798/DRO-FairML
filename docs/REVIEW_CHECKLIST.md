# DRO-FAIR — Self-Review Checklist

A structured checklist for reviewing the project before submission or external review. Each category lists the questions a rigorous reviewer would ask, the code locations involved, and the criterion for a passing answer.

This is organized into nine areas. Work through them in order — earlier categories are prerequisites for later ones.

---

## 1. Algorithm Verification

The following questions should be answerable from memory, with derivations on paper.

- Walk through Algorithm 1 line by line. Why is the θ update before the λ and p updates?
- Derive the DP radius `ρ_j = α / ((1 − α) · π_j + α)` from Theorem 4.2.
- Derive the bias correction `π_clean = (π_obs − α) / (1 − 2α)` from first principles.
- Why is the inner gradient `∇g` and not `λ∇g`? Show that the argmax is unchanged.
- Explain Dykstra's alternating projection convergence onto `Δ_n ∩ B_1`.
- What does the tilted loss with `β = 5` look like at `β → ∞` and `β → 0`? Why 5?
- Why is the L1-ball radius `2ρ` and not `ρ`?
- Why the τ warmup? What happens without it?

**Criterion:** Every question answerable from first principles, without consulting code or notes.

---

## 2. Code Audit

Read each file completely and check the criteria below.

**Files in scope:**
- [src/training/dro_fair.py](../src/training/dro_fair.py)
- [src/training/naive_fair.py](../src/training/naive_fair.py)
- [src/corruption/adversarial.py](../src/corruption/adversarial.py)
- [src/utils/projections.py](../src/utils/projections.py)
- [src/evaluation/metrics.py](../src/evaluation/metrics.py)

**Checks per file:**
- Syntax: no import errors, no undefined variables
- Numerical safety: no NaN/Inf paths (guards on divisions, clipping where needed)
- Paper match: matches Algorithm 1 exactly (step order, gradient form, radii)
- Edge cases: empty groups, α = 0, α = 0.5 boundary, single-class batches
- Comments: explain *why* a non-obvious choice was made, not *what* the code does

**Specific invariants in `dro_fair.py`:**
- Step order θ → λ → p
- Inner gradient `∇g` (not `λ∇g`)
- Bias-corrected radii using Appendix F formula
- Dykstra tail `max_iter = 500`
- L1-ball radius `2 * rho` (not `rho`)

**Specific invariants in `adversarial.py`:**
- PGD loop with sign-based perturbation and ε-projection
- Label attack maximizes group disparity
- Attribute attack respects the 70%-minority targeting
- `RandomCorruptor` is a distinct class with random (not gradient-based) attacks

---

## 3. Threat Model Review

The adversarial setting must be precisely defined and consistent with theory.

- Define the threat model precisely: who is the attacker, what do they know, what can they modify?
- PGD vs Carlini-Wagner: why PGD?
- Coordinated label/attribute flips: under whose control? Within the same αn budget?
- Theorem 6.1 is stated for α-corruption. Does coordinated PGD still respect the TV-ball assumption?
- Where do the DRO guarantees become loose under coordinated multi-modal adversarial corruption?

**Criterion:** Each question has a one-paragraph answer that ties back to the radii calibration in Theorems 4.2 and 4.3.

---

## 4. Results Audit

Verify only against the saved data files:
- `results/all_results.json`
- `results/individual/*.json`

**Checks:**
- 150 experiments complete (no missing seeds)
- No NaN or Inf in any metric
- Means and standard errors computed correctly
- DRO wins at α ∈ {0.1, 0.2, 0.3} satisfy the Wilcoxon p < 0.05 criterion in ≥ 6/9 cells for DP
- α = 0.0 cells are ties (DRO radii collapse to zero)
- Credit α = 0.4 DRO accuracy ≥ 0.70
- No prediction collapse (no row where accuracy < 0.5 unless documented as the Adult failure mode)

**Statistical significance (critical):**

Run Wilcoxon signed-rank for each (dataset, α) cell:
```python
from scipy.stats import wilcoxon
stat, p = wilcoxon(naive_dps, dro_dps, alternative='greater')
# "DRO significantly better" requires p < 0.05
```

Cells where the mean favors DRO but p ≥ 0.05 must be reported as "insufficient evidence", not as wins.

---

## 5. Deliverables Audit

For submission acceptance, all of the following must exist and be valid:

- [ ] **Table 1.** 3 datasets × 5 α, mean ± SE over 10 seeds. Columns: accuracy, DP violation, IF violation for both Naive and DRO. Values within reasonable ranges.
- [ ] **Ablations.** DP-only vs DP+IF, random vs adversarial corruption, runtime overhead.
- [ ] **Theory verification.** Output of `verify_theory.py` showing all four theorems pass on real data.
- [ ] **Figures.** DP-vs-α and IF-vs-α curves per dataset; runtime overhead.
- [ ] **Report.** Problem formulation, methodology, results, discussion (with limitations), related work.
- [ ] **Tests.** Full test suite passes (32 tests across 4 files).

---

## 6. Author Understanding

The following six questions should be answerable in conversation, without referencing the code or report.

- D1: Walk through Algorithm 1. Why is θ before λ before p? What happens if p updates first?
- D2: Derive `ρ_j = α / ((1 − α) · π_j + α)` from the TV distance definition.
- D3: Derive the bias correction `(π_obs − α) / (1 − 2α)`.
- D4: Explain Dykstra's alternating projection and why it converges to the intersection.
- D5: Explain PGD step by step. What is the sign doing? What is the projection preventing?
- D6: Explain the tilted loss `β · (logsumexp(ℓ / β) − log(m))` in the limits `β → ∞` and `β → 0`.

**Criterion:** 6/6 confident answers indicates readiness. 4–5 indicates additional preparation is warranted. ≤ 3 indicates the project cannot be defended in its current form.

---

## 7. Novelty Assessment

Be precise about the contribution.

**Stated contribution.** We extend the paper's random-noise evaluation to a multi-modal adversarial corruption protocol and characterize where the DRO guarantee remains tight (Credit, LSAC) versus where it over-corrects (Adult).

**Self-questions:**
- What new empirical insight does the adversarial setting reveal that random noise does not?
- Do the paper's theorems still apply, weaken, or break? Where exactly?
- Is there a formal analysis (even informal) of the coordinated multi-modal threat versus per-modality radii?

**Criterion:** The contribution should be one paragraph and should not reduce to "we re-ran the paper with a different noise process."

---

## 8. Repository Cleanliness

The following types of files should not exist in the submitted repo:
- Ad-hoc orchestration scripts (`start_all.py`, `stop_all.py`, `monitor_*.py`, `fix_stale_files.py`)
- AI/agent conversation transcripts (`session-*.md`, vendor exports)
- Duplicate experiment runners that conflict with each other

**Files expected to exist:**
- `src/`, `tests/`, `configs/`, `data/`, `results/`, `figures/`
- `experiments/`: run_experiments.py, run_robust.py, run_ablations.py, run_random_vs_adversarial.py, generate_results.py, generate_figures.py, validate_results.py, verify_theory.py, diagnostics.py
- `main.py`, `setup.py`, `requirements.txt`, `README.md`, `Makefile`
- `report/report.tex`, `report/report.pdf`
- `submission/` with frozen artifacts

---

## 9. Hyperparameter Audit

Every hyperparameter requires a one-line justification.

| Parameter | Value | Source |
|-----------|-------|--------|
| `lambda_max` | 1.5 | Stability adjustment from paper's 2.0 (prevents Adult collapse at α = 0.2) |
| `epochs` | 60 | Paper §7.1 |
| `K_inner` | 10 | Paper §G.4 |
| `lr_theta` | 1e-3 | Paper §G.4 |
| `lr_lambda` | 5e-3 | Paper §G.4 |
| `lr_p` | 5e-3 | Paper §G.4 |
| `beta` | 5.0 | Paper §G.5 |
| `tau` | 100 / 1 (schedule) | Paper §G.6 |
| `tau_warmup_epochs` | 15 | Stability (paper uses 10) |
| `lambda_lr_decay` | 0.95^epoch | Stability, prevents λ overgrowth |
| `grad_clip` | 0.5 | Stability (paper uses 1.0) |
| `pgd_epsilon` | 0.1 | Standard PGD choice |

Any parameter without a clear source in either the paper or a documented stability fix should be marked as needing justification.

---

## Grading Rubric

| Category | Weight | A criterion |
|----------|--------|-------------|
| Algorithm correctness | 20% | All invariants in §1 and §2 hold |
| Results validity | 25% | All checks in §4 pass, statistical significance verified |
| Novelty / contribution | 15% | One-paragraph contribution distinct from re-running the paper |
| Author understanding | 15% | 6/6 in §6 |
| Deliverables | 15% | All boxes in §5 checked |
| Repository cleanliness | 10% | All conditions in §8 satisfied |

Total grade is the weighted average. A ≥ 85, B ≥ 70, C ≥ 50.

---

## Output Format

When using this checklist for a structured review, produce a short report containing:

1. **Verdict.** Accept / Revise / Reject.
2. **Per-category scores** with one-line notes.
3. **Critical failures** that must be addressed (with file:line references).
4. **Minor issues** that should be addressed.
5. **Understanding-check questions** asked and answered, with verdict per question.
6. **Files flagged for deletion** (if any).
7. **Missing deliverables** (if any).
8. **Recommended priority order** for fixes.

---

## Notes on Conducting the Review

- Demand proofs and references, not assertions. "It's in the paper" is acceptable when matching the paper; it is not acceptable when justifying a choice that deviates from the paper.
- Check reproducibility: random seeds are fixed everywhere, the experiment runner is deterministic, and CI verifies on a clean checkout.
- Verify statistics: SEs are computed correctly, error bars are reasonable, sample size (n = 10) is sufficient for the claims made.
- Look for the kind of small inconsistencies that indicate the work has not been audited: numbers that differ between report and CSV, captions that cite a test that was not run, claims that the data does not support.
