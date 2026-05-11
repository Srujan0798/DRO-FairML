# Agent Guide for DRO-FAIR Project

## Project Overview

This project implements DRO-FAIR (2nd approach from ICML submission) with a key modification: **adversarial noise instead of random noise**. The codebase is fully PyTorch-based and designed for reproducible fairness research under data corruption.

## Architecture

```
src/
├── data/datasets.py        # Adult, Credit, LSAC loaders + synthetic fallback
├── models/classifier.py    # Simple MLP with dropout
├── corruption/adversarial.py  # PGD/FGSM features + coordinated label/attr flips
├── training/
│   ├── naive_fair.py      # Baseline: fairness on corrupted data (ρ=0)
│   └── dro_fair.py        # Algorithm 1: min-max with Dykstra projection
├── evaluation/metrics.py   # Accuracy, DP, IF violations
└── utils/projections.py    # Simplex + L1-ball projection (Dykstra)
```

## Key Design Decisions

1. **Synthetic Data Fallback**: Due to SSL certificate issues on the host system, datasets fall back to realistic synthetic data with group-dependent distributions when downloads fail.

2. **Vectorized IF Computation**: The k-NN IF loss is computed via precomputed edge tensors rather than nested Python loops, providing ~40× speedup.

3. **Checkpointing**: The experiment runner saves progress every 5 experiments to `results/checkpoint.pkl` and resumes automatically.

## Running Experiments

```bash
# Full pipeline (experiments + results)
python main.py --full-pipeline

# Just experiments
python experiments/run_experiments.py

# Quick demo
python scripts/demo.py --dataset adult --alpha 0.2
```

## Testing

```bash
pytest tests/ -v
```

## Hyperparameters (from paper Section 7.1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ | 100 | Temperature for soft classifier |
| β | 5 | Tilting parameter for CVaR approximation |
| k | 5 | k-NN neighbors for IF |
| K_inner | 10 | Inner maximization steps |
| η_θ | 1e-3 | Learning rate for model |
| η_λ | 5e-3 | Learning rate for Lagrange multipliers |
| η_p | 1e-2 | Learning rate for importance weights |

## Common Issues

1. **SSL Download Failures**: System Python 3.14 lacks proper SSL certs. Code falls back to synthetic data automatically.
2. **Memory**: Full experiments use ~1.5GB RAM. Reduce batch size or use smaller hidden dims if needed.
3. **Speed**: DRO-FAIR is ~12× slower than Naive-FAIR due to inner maximization loop.

## Extending the Code

- **New datasets**: Add loader to `src/data/datasets.py`, update `get_dataset()`
- **New corruption types**: Extend `AdversarialCorruptor` in `src/corruption/adversarial.py`
- **New fairness metrics**: Add to `src/evaluation/metrics.py`
- **Model architectures**: Modify `MLPClassifier` or create new class in `src/models/`
