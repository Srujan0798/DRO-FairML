# DRO-FAIR Frequently Asked Questions

## Q1: Why does Adult fail at high α?
**A:** Adult has baseline DP ~0.17 (8× larger than Credit/LSAC). Coordinated adversarial label flips amplify this disparity. DRO's conservative TV radii trigger λ_DP runaway, collapsing the model to near-uniform predictions. This is documented in report Section 5 — it's an honest empirical limitation, not a code bug.

## Q2: Is 5/9 IF worse than 7/9 IF?
**A:** Both are correct under different definitions. Mean-based: 7/9. Wilcoxon p<0.05: 5/9. The report explicitly claims "Wilcoxon p<0.05" in figure captions, so 5/9 is the only internally consistent answer. Claiming 7/9 while citing Wilcoxon would be a methodological inconsistency.

## Q3: Why is lambda_max=1.5 instead of the paper's 2.0?
**A:** Stability fix. With λ_max=2.0, λ_DP runaway on Adult caused model collapse even at α=0.2. Reducing to 1.5 caps the penalty and prevents collapse while preserving wins on Credit/LSAC.

## Q4: Why is runtime 37.5× and not 12× like the paper?
**A:** The paper reports ~12× on GPU. Our implementation runs on CPU with full-batch k-NN graph construction per epoch, which dominates overhead. GPU training would reduce this significantly.

## Q5: Can I reproduce the results?
**A:** Yes. All random seeds are fixed (0–9). Run:
```bash
python3 experiments/run_experiments.py --n_seeds 10
python3 experiments/generate_figures.py
python3 experiments/verify_theory.py
```

## Q6: Why no GPU support?
**A:** The code uses `device='cpu'` by default but will use CUDA if available. However, the k-NN graph construction (sklearn NearestNeighbors) runs on CPU regardless, so GPU speedup is limited.

## Q7: Is the adversarial threat model realistic?
**A:** It's a white-box threat model where the attacker controls α-fraction of samples and can modify features, labels, and attributes. This is standard in adversarial ML literature. The report discusses limitations: multi-modal attacks concentrate signal in ways per-modality radii didn't anticipate.
