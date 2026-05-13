# Agent Guide for DRO-FAIR Project

## Project Overview

This project implements DRO-FAIR (2nd approach from ICML submission) with **adversarial noise instead of random noise**. The codebase is fully PyTorch-based and designed for reproducible fairness research under data corruption.

**Key fact: All datasets are REAL.** LSAC is downloaded from `damtharvey/law-school-dataset` (18,692 rows). No synthetic fallback is used.

## Architecture

```
src/
├── data/datasets.py           # Adult, Credit, LSAC loaders (ALL REAL DATA)
├── models/classifier.py       # Simple MLP with dropout
├── corruption/adversarial.py  # AdversarialCorruptor + RandomCorruptor
├── training/
│   ├── naive_fair.py         # Baseline: fairness on corrupted data (ρ=0)
│   ├── dro_fair.py           # Algorithm 1: exact paper implementation
│   └── standard_ml.py        # Standard ML (no fairness constraints)
├── evaluation/metrics.py     # Accuracy, DP, IF violations
└── utils/projections.py      # Simplex + L1-ball projection (Dykstra)
```

## Key Design Decisions

1. **Real Data Only**: LSAC downloaded via `curl` from GitHub (18,692 rows). Adult and Credit also use real data.
2. **Exact Algorithm 1**: K_inner=10, exact tilted risk β·log(mean(exp(ℓ/β))), IF scaling divides by (n−1), Dykstra projection
3. **Temperature Tuning**: τ=100 for α≤0.3, τ=1 for α≥0.4 (auto-configured in DroFairTrainer)
4. **Test-Time Evaluation**: Both clean and corrupted test evaluation
5. **Checkpointing**: Experiment runner saves progress every 5 experiments to `results/checkpoint.pkl`

## Running Experiments

```bash
# Full pipeline (experiments + results)
python main.py --full-pipeline

# Just experiments
python experiments/run_experiments.py

# Random vs adversarial comparison
python experiments/run_random_vs_adversarial.py

# Verify theory
python experiments/verify_theory.py

# Quick demo
python scripts/demo.py --dataset adult --alpha 0.2
```

## Hyperparameters (Exact from Paper Section 7.1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ | auto (100 or 1) | Temperature for soft classifier |
| β | 5 | Tilting parameter for CVaR approximation |
| k | 5 | k-NN neighbors for IF |
| K_inner | 10 | Inner maximization steps |
| η_θ | 1e-3 | Learning rate for model |
| η_λ | 5e-3 | Learning rate for Lagrange multipliers |
| η_p | 5e-3 | Learning rate for importance weights |
| λ_max | 10.0 | Max Lagrange multiplier |
| epochs | 30 | Training epochs |

## Common Issues

1. **SSL Download Failures**: System Python 3.14 lacks proper SSL certs. Code uses `curl` fallback for Adult/Credit and direct CSV download for LSAC.
2. **Memory**: Full experiments use ~1.5GB RAM.
3. **Speed**: DRO-FAIR is ~2× slower than Naive-FAIR on CPU. Paper reports ~12× on GPU with larger models.

## Testing

```bash
pytest tests/ -v
```

23 tests covering projections, corruption, end-to-end training, algorithm correctness, and data integrity.

## Extending the Code

- **New datasets**: Add loader to `src/data/datasets.py`, update `get_dataset()`
- **New corruption types**: Extend `AdversarialCorruptor` or add `RandomCorruptor` variants
- **New fairness metrics**: Add to `src/evaluation/metrics.py`
- **Model architectures**: Modify `MLPClassifier` or create new class
