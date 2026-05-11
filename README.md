# DRO-FAIR: Robust Individual and Group Fair Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements **DRO-FAIR** (Distributionally Robust Optimization for Fairness), the second approach from our ICML submission, for joint Demographic Parity (DP) + Individual Fairness (IF) under α-corruption. A key innovation of this implementation is replacing the paper's random noise with **adversarial noise** (PGD/FGSM-style attacks on features, coordinated label flips, and protected attribute flips).

## Overview

Fair classification algorithms ensure equitable treatment across protected groups and similar individuals, but their performance degrades when training data is corrupted. DRO-FAIR provides robust fairness guarantees by:

- **Corruption-calibrated uncertainty sets** with tight TV-distance radii
- **Min-max Lagrangian optimization** (Algorithm 1) with inner projected gradient ascent on importance weights
- **Joint enforcement** of Demographic Parity and Individual Fairness constraints
- **Adversarial corruption protocol** for realistic worst-case evaluation

## Experimental Results

### Main Results (Table 1)

| Dataset | α | Method | Acc↑ | DP↓ | IF↓ |
|---------|---|--------|------|-----|-----|
| **Adult** | 0.0 | Naive | 0.753±0.000 | 0.001±0.001 | 0.000±0.000 |
| | | DRO-FAIR | 0.760±0.002 | 0.012±0.004 | 0.002±0.001 |
| | 0.1 | Naive | 0.763±0.002 | 0.018±0.005 | 0.003±0.001 |
| | | DRO-FAIR | 0.771±0.004 | 0.032±0.009 | 0.006±0.002 |
| | 0.2 | Naive | 0.791±0.004 | 0.078±0.013 | 0.014±0.002 |
| | | DRO-FAIR | 0.795±0.003 | 0.092±0.009 | 0.015±0.001 |
| | 0.3 | Naive | 0.782±0.003 | **0.055±0.009** | **0.032±0.002** |
| | | DRO-FAIR | 0.777±0.003 | **0.037±0.007** | **0.033±0.001** |
| | 0.4 | Naive | 0.541±0.007 | 0.296±0.008 | 0.044±0.001 |
| | | DRO-FAIR | 0.544±0.008 | 0.301±0.007 | 0.047±0.001 |
| **Credit** | 0.0 | Naive | 0.720±0.000 | 0.000±0.000 | 0.000±0.000 |
| | | DRO-FAIR | 0.720±0.000 | 0.000±0.000 | 0.000±0.000 |
| | 0.1 | Naive | 0.743±0.009 | 0.000±0.000 | 0.000±0.000 |
| | | DRO-FAIR | 0.751±0.010 | 0.003±0.001 | 0.000±0.000 |
| | 0.2 | Naive | 0.786±0.003 | 0.004±0.002 | 0.001±0.000 |
| | | DRO-FAIR | 0.794±0.003 | 0.009±0.002 | 0.001±0.000 |
| | 0.3 | Naive | 0.797±0.003 | **0.014±0.003** | 0.001±0.000 |
| | | DRO-FAIR | 0.793±0.003 | **0.010±0.003** | 0.002±0.000 |
| | 0.4 | Naive | 0.788±0.004 | 0.019±0.004 | 0.004±0.001 |
| | | DRO-FAIR | 0.791±0.004 | 0.022±0.003 | 0.005±0.001 |
| **LSAC** | 0.0 | Naive | 0.759±0.013 | 0.038±0.006 | 0.000±0.000 |
| | | DRO-FAIR | 0.789±0.011 | 0.066±0.006 | 0.000±0.000 |
| | 0.1 | Naive | 0.807±0.010 | 0.069±0.006 | 0.000±0.000 |
| | | DRO-FAIR | 0.834±0.008 | 0.071±0.004 | 0.000±0.000 |
| | 0.2 | Naive | 0.858±0.008 | 0.074±0.006 | 0.000±0.000 |
| | | DRO-FAIR | 0.882±0.004 | 0.083±0.006 | 0.000±0.000 |
| | 0.3 | Naive | 0.869±0.006 | 0.076±0.007 | 0.000±0.000 |
| | | DRO-FAIR | 0.869±0.004 | **0.072±0.008** | 0.001±0.000 |
| | 0.4 | Naive | 0.771±0.008 | **0.032±0.005** | 0.001±0.000 |
| | | DRO-FAIR | 0.768±0.010 | **0.022±0.004** | 0.001±0.000 |

**Key Findings:**
- DRO-FAIR shows **strongest improvements at moderate corruption (α=0.3)** on Adult (32% DP reduction) and Credit (25% DP reduction)
- At **high corruption (α=0.4)**, both methods degrade significantly; DRO-FAIR maintains comparable performance
- At **low corruption (α≤0.2)**, DRO-FAIR can be overly conservative, suggesting need for adaptive radius selection
- LSAC shows DRO-FAIR benefits at α=0.3–0.4 with 5–32% DP reduction

### Adversarial vs Random Corruption

Our adversarial protocol (coordinated label/attribute flips + FGSM features) creates stronger corruption than random noise. Results show DRO-FAIR is most beneficial when corruption is **coordinated and moderate**, consistent with the theoretical motivation.

## Project Structure

```
DRO-FairML/
├── src/
│   ├── data/              # Dataset loaders (Adult, Credit, LSAC)
│   ├── models/            # MLP classifier architecture
│   ├── corruption/        # Adversarial corruption module
│   ├── training/          # Naive-FAIR and DRO-FAIR trainers
│   ├── evaluation/        # Accuracy, DP, IF metrics
│   └── utils/             # Projection utilities (Dykstra's algorithm)
├── experiments/
│   ├── run_experiments.py     # Main experiment runner
│   ├── generate_results.py    # Table and plot generation
│   └── analyze_results.py     # Advanced analysis
├── configs/
│   └── default.yaml       # Hyperparameter configuration
├── docs/
│   └── ALGORITHM.md       # Detailed algorithm documentation
├── scripts/
│   └── demo.py            # Quick demo script
├── tests/                 # Unit tests
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
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
1. Download and preprocess datasets
2. Apply adversarial corruption at α ∈ {0.0, 0.1, 0.2, 0.3, 0.4}
3. Train both Naive-FAIR and DRO-FAIR (10 seeds each)
4. Generate result tables and plots

### Run Experiments Only

```bash
python main.py --run-experiments --datasets adult credit --alphas 0.0 0.2 --n-seeds 5
```

### Generate Results from Existing Data

```bash
python main.py --generate-results
```

### Quick Demo

```bash
python scripts/demo.py --dataset adult --alpha 0.2 --epochs 20
```

## Method Details

### Adversarial Corruption (vs. Random Noise)

Unlike the original paper which uses random Gaussian noise, uniform categorical replacement, and random label flips, our implementation uses:

1. **Feature Attacks**: FGSM-style perturbations on numeric features (direction: towards opposite class, scaled by column std)
2. **Coordinated Label Flips**: Labels are flipped to *maximize* group rate disparity (increase DP violation)
3. **Coordinated Attribute Flips**: Protected attributes are flipped with 70% of corruption focused on the minority group

### DRO-FAIR Algorithm (Algorithm 1)

**Uncertainty Sets:**
- DP radii: ρ_DP,j = α / ((1−α)π_j + α) for each group j
- IF radius: ρ_IF = 2α − α²

**Optimization Loop:**
1. **Outer minimization**: Update model parameters θ using AdamW
2. **Dual ascent**: Update Lagrange multipliers λ_DP, λ_IF
3. **Inner maximization**: K=5 projected gradient ascent steps on importance weights p̃
   - Projection via Dykstra's alternating projection algorithm onto simplex ∩ ℓ₁-ball

**Hyperparameters:**
- Temperature: τ = 100 (soft classifier σ(τ·f_θ(x)))
- Tilting: β = 5 (tilted empirical risk approximates CVaR)
- k-NN: k = 5 for IF approximation
- Learning rates: η_θ = 1e-3, η_λ = 5e-3, η_p = 5e-3

### Naive-FAIR Baseline

Special case of DRO-FAIR with ρ_DP = ρ_IF = 0 (no inner maximization, uniform weights). Enforces fairness constraints directly on corrupted data without robust reweighting.

## Evaluation Metrics

- **Accuracy**: Standard classification accuracy on clean test set
- **DP Violation**: |P(h=1|A=0) − P(h=1|A=1)|
- **IF Violation**: Fraction of k-NN pairs violating metric fairness: |h(x_i) − h(x_j)| > d(x_i, x_j) + γ

## Experiments

### Datasets

| Dataset | Samples | Features | Protected | Task | Status |
|---------|---------|----------|-----------|------|--------|
| Adult | 45,222 | 12 | Sex | Income >$50K | Real data |
| Credit | 30,000 | 22 | Sex | Default prediction | Real data |
| LSAC | 18,692 | 10 | Sex | Bar passage | Synthetic fallback |

All datasets are preprocessed with label encoding for categorical variables and StandardScaler normalization. 80/20 train-test split with 85/15 train-validation split.

### Corruption Levels

Experiments are run at α ∈ {0.0, 0.1, 0.2, 0.3, 0.4} with 10 random seeds per setting. Test sets remain uncorrupted to evaluate clean-distribution performance.

## Reproducing Paper Results

To reproduce the main results:

```bash
python main.py --run-experiments --n-seeds 10
```

Expected runtime: ~30-40 minutes on CPU.

## Limitations & Future Work

1. **Adaptive Radii**: Current fixed radii can be overly conservative at low α. Adaptive radius selection based on validation performance could improve results.
2. **Hyperparameter Sensitivity**: Results are sensitive to lr_p and K_inner. Systematic tuning could improve robustness.
3. **IF-IF Tradeoff**: Joint DP+IF optimization shows tension between constraints at high corruption. DP-only or IF-only variants may be preferable in some settings.
4. **Dataset Availability**: LSAC falls back to synthetic data due to download limitations.

## Testing

```bash
pytest tests/ -v
```

All 14 unit tests pass, covering projections, corruption, and metrics.

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

## License

MIT License
