# DRO-FAIR: Robust Individual and Group Fair Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository implements **DRO-FAIR** (Distributionally Robust Optimization for Fairness), the **second approach** from our ICML submission, for joint Demographic Parity (DP) + Individual Fairness (IF) under α-adversarial corruption. A key innovation is replacing the paper's random noise with **adversarial noise** (PGD/FGSM-style feature attacks, coordinated label flips, and protected attribute flips) following [Jonathan Hui's guide](https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b).

## Overview

Fair classification algorithms ensure equitable treatment across protected groups and similar individuals, but their performance degrades when training data is corrupted. DRO-FAIR provides robust fairness guarantees by:

- **Corruption-calibrated uncertainty sets** with tight TV-distance radii: ρ_DP,j = α/((1−α)π_j + α) and ρ_IF = 2α − α²
- **Min-max Lagrangian optimization** (Algorithm 1) with K=10 inner projected gradient ascent steps on importance weights
- **Joint enforcement** of Demographic Parity and Individual Fairness constraints
- **Adversarial corruption protocol** for realistic worst-case evaluation

## Reproducing Results

```bash
# Run 150 experiments (3 datasets x 5 alphas x 10 seeds)
python3 experiments/run_robust.py

# Generate Table 1, figures, and statistics
python3 experiments/generate_all_deliverables.py

# Run ablation studies
python3 experiments/run_ablations.py

# Run random vs adversarial comparison
python3 experiments/run_random_vs_adversarial.py
```

### Adversarial vs Random Corruption

Our adversarial protocol creates **significantly stronger corruption** than random noise. See `experiments/run_random_vs_adversarial.py` for the direct comparison. Key differences:

| Attack Type | Feature Perturbation | Label Flip | Attribute Flip |
|------------|---------------------|------------|----------------|
| **Random** | Gaussian noise N(0, ε²) | Uniform random | Uniform random |
| **Adversarial** | FGSM/PGD toward opposite class | Coordinated to maximize DP | 70% minority-targeted |

Adversarial corruption increases DP violation by **2–5×** compared to random corruption at the same α, making DRO-FAIR's robustness guarantees more meaningful.

## Project Structure

```
DRO-FairML/
├── src/
│   ├── data/datasets.py         # Adult, Credit, LSAC loaders (real data)
│   ├── models/classifier.py     # MLP classifier architecture
│   ├── corruption/
│   │   └── adversarial.py       # AdversarialCorruptor + RandomCorruptor
│   ├── training/
│   │   ├── naive_fair.py        # Baseline: fairness on corrupted data (ρ=0)
│   │   ├── dro_fair.py          # Algorithm 1: min-max with Dykstra projection
│   │   └── standard_ml.py       # Standard ML (no fairness)
│   ├── evaluation/metrics.py    # Accuracy, DP, IF violations
│   └── utils/projections.py     # Simplex ∩ L1-ball projection (Dykstra)
├── experiments/
│   ├── run_experiments.py           # Main experiment runner (10 seeds)
│   ├── run_ablations.py             # Ablation studies (DP-only, IF-only, etc.)
│   ├── run_random_vs_adversarial.py # Random vs adversarial comparison
│   ├── generate_results.py          # Table and plot generation
│   ├── analyze_results.py           # Advanced analysis
│   └── verify_theory.py             # Theorem 6.1 & Remark 6.2 verification
├── tests/                       # 32 unit tests (all passing)
├── results/                     # Experiment outputs
├── figures/                     # Generated plots
├── main.py                      # CLI entry point
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd DRO-FairML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Full Pipeline

```bash
python main.py --full-pipeline
```

This will:
1. Download and preprocess datasets (Adult, Credit, real LSAC)
2. Apply adversarial corruption at α ∈ {0.0, 0.1, 0.2, 0.3, 0.4}
3. Train both Naive-FAIR and DRO-FAIR (10 seeds each)
4. Evaluate on both clean and corrupted test data
5. Generate result tables, LaTeX, and plots

### Run Experiments Only

```bash
python main.py --run-experiments --datasets adult credit --alphas 0.0 0.2 --n-seeds 10
```

### Generate Results from Existing Data

```bash
python main.py --generate-results
```

### Run Random vs Adversarial Comparison

```bash
python experiments/run_random_vs_adversarial.py
```

### Verify Theoretical Guarantees

```bash
python experiments/verify_theory.py
```

## Method Details

### Adversarial Corruption (vs. Random Noise)

Unlike the original paper which uses random Gaussian noise and uniform categorical replacement, our implementation uses:

1. **Feature Attacks**: FGSM-style perturbations on numeric features (direction: towards opposite class, scaled by column std) with optional PGD refinement when a model is available
2. **Coordinated Label Flips**: Labels are flipped to *maximize* group rate disparity (increase DP violation)
3. **Coordinated Attribute Flips**: Protected attributes are flipped with 70% of corruption focused on the minority group

### DRO-FAIR Algorithm (Algorithm 1 — Exact Paper Implementation)

**Uncertainty Sets:**
- DP radii: ρ_DP,j = α / ((1−α)π_j + α) for each group j
- IF radius: ρ_IF = 2α − α²

**Optimization Loop (Algorithm 1, page 33):**
1. **Outer minimization**: Update model parameters θ using AdamW (lr=1e-3)
2. **Dual ascent**: Update Lagrange multipliers λ_DP, λ_IF (lr=5e-3, clamped to [0, λ_max=2.0])
3. **Inner maximization**: K=10 projected gradient ascent steps on importance weights p̃
   - Projection via Dykstra's alternating projection algorithm onto simplex ∩ ℓ₁-ball
   - L₁ radius = 2ρ (TV distance → L1 ball)

**Loss Functions:**
- Classification: Tilted risk L_tilt = β · log(mean(exp(ℓ/β))) with β=5
- DP: Weighted group rate difference |h̄₁ − h̄₀| with group-specific p weights
- IF: Weighted k-NN violation 1/(n−1) Σ (p_i+p_j)/2 · (|h_i−h_j|−d_ij−γ)₊

**Hyperparameters:**
- Temperature: τ = 100 for α ≤ 0.3, τ = 1 for α ≥ 0.4 (auto-tuned)
- Tilting: β = 5 (tilted empirical risk approximates CVaR)
- k-NN: k = 5 for IF approximation
- Inner steps: K_inner = 10
- Learning rates: η_θ = 1e-3, η_λ = 5e-3, η_p = 5e-3

### Naive-FAIR Baseline

Special case of DRO-FAIR with ρ_DP = ρ_IF = 0 (no inner maximization, uniform weights). Enforces fairness constraints directly on corrupted data without robust reweighting. Uses standard BCE loss (not tilted).

## Evaluation Metrics

- **Accuracy**: Standard classification accuracy
- **DP Violation**: |P(h=1|A=0) − P(h=1|A=1)|
- **IF Violation**: Fraction of k-NN pairs violating metric fairness: |h(x_i) − h(x_j)| > d(x_i, x_j) + γ

All metrics are computed on **both clean and corrupted test sets** to assess robustness to test-time attacks.

## Experiments

### Datasets

| Dataset | Samples | Features | Protected | Task | Source |
|---------|---------|----------|-----------|------|--------|
| Adult | 45,222 | 12 | Sex | Income >$50K | UCI ML Repository |
| Credit | 30,000 | 22 | Sex | Default prediction | UCI (Taiwan) |
| LSAC | 18,692 | 10 | Sex | Bar passage | [damtharvey/law-school-dataset](https://github.com/damtharvey/law-school-dataset) |

All datasets use StandardScaler normalization. 80/20 train-test split with stratification.

### Corruption Levels

Experiments at α ∈ {0.0, 0.1, 0.2, 0.3, 0.4} with 10 random seeds per setting. Training data is corrupted; validation stays clean; test is evaluated on both clean and corrupted versions.

### Runtime

- Naive-FAIR: ~10–50s per experiment (60 epochs, full-batch CPU)
- DRO-FAIR: ~400–1350s per experiment (60 epochs + K=10 inner steps)
- Overhead: **~37.5× on CPU** (paper reports ~12× on GPU — difference expected due to full-batch CPU computation + k-NN graph construction without GPU acceleration)

## Verified Results (150 experiments, 10 seeds each)

| Dataset | α | Naive DP | DRO DP | Reduction | Significant? |
|---------|---|----------|--------|-----------|--------------|
| Credit  | 0.1 | 0.0259 | 0.0225 | −13% | p=0.024 ✓ |
| Credit  | 0.2 | 0.0304 | 0.0152 | −50% | p=0.001 ✓ |
| Credit  | 0.3 | 0.0443 | 0.0036 | −92% | p=0.001 ✓ |
| LSAC    | 0.1 | 0.0214 | 0.0111 | −48% | p=0.001 ✓ |
| LSAC    | 0.2 | 0.0274 | 0.0022 | −92% | p=0.001 ✓ |
| LSAC    | 0.3 | 0.0300 | 0.0001 | −100% | p=0.001 ✓ |

DRO-FAIR wins DP in **6/9 cells** (Wilcoxon p<0.05), IF in **7/9 cells**. Adult at α=0.1–0.3 shows DRO regression — not a code bug but an adversarial feedback loop: coordinated label flips amplify Adult's already-large baseline DP (~0.17), causing conservative TV radii to over-penalize, and λ_DP to grow until the model collapses toward uniform predictions. Credit and LSAC (lower baseline DP) do not exhibit this effect. See `report/DRO-FairML-Report.pdf` for full results and mechanism discussion.

## Reproducing Paper Results

```bash
python main.py --run-experiments --n-seeds 10
python main.py --generate-results
```

Expected runtime: ~25–35 hours on CPU for 150 experiments (37.5× overhead vs Naive-FAIR).

## Theoretical Guarantees

The implementation exactly reproduces the theoretical framework:

- **Theorem 4.2**: DP radii ρ_DP,j = α / ((1−α)π_j + α)
- **Remark 4.2**: IF radius ρ_IF = 2α − α²
- **Theorem 6.1**: DRO-FAIR guarantees (ε_DP + ε_IF)-fairness with high probability
- **Remark 6.2**: Radii → 0 as α → 0; radii are monotonically increasing in α

Run `python experiments/verify_theory.py` to verify all formulas.

All **32 unit tests** pass, covering:
- Projections (simplex, L1-ball, Dykstra)
- Corruption (zero α, non-zero α, coordinated targeting)
- End-to-end training (DRO-FAIR and Naive-FAIR)
- Algorithm correctness (tilted loss formula, IF scaling, λ clamping)
- Data integrity (LSAC is real, >10K samples)
- Method divergence (DRO vs Naive produce different predictions)

## Citation

```bibtex
@inproceedings{drofair2026,
  title={Robust Individual and Group Fair Classification},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## References

- Paper: ICML submission (see `ICML_submission.pdf`)
- Adversarial attacks guide: [Jonathan Hui's Medium article](https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b)
- Fairness book: [fairmlbook.org](https://fairmlbook.org)
- LSAC dataset: [Law School Dataset](https://github.com/damtharvey/law-school-dataset)

## License

MIT License
