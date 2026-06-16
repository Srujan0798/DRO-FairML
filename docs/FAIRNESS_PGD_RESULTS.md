# FAIRNESS_PGD_RESULTS — Week 2 (Updated: tau=1 era)

## Setup

**Task:** Test whether DRO-FAIR is more robust than Naive-FAIR under adversarial fairness attacks.

**Datasets:** Adult (UCI), Credit Default, LSAC Bar Passage

**Attack modes:**
- **DP-attack:** Compute gradient of Demographic Parity w.r.t. each label, flip labels that increase DP violation most
- **IF-attack:** Compute gradient of Individual Fairness using k-NN agreement within protected groups, flip labels that increase IF violation most
- **Combined:** Equal weighted sum of DP and IF gradients

**Methods:** Naive-FAIR (standard Lagrangian), DRO-FAIR (corruption-calibrated uncertainty sets)

**Protocol:** 3 seeds per condition (n=6 in progress). Prediction temperature fixed at tau=1 (the tau=100 stepped schedule was the source of earlier "DRO is fragile" finding).

---

## Headline Result: tau=1 makes DRO beat Naive on Adult DP at every alpha

| Attack | alpha | Naive DP | DRO DP | DRO wins/3 seeds | delta acc |
|--------|-------|----------|--------|-------------------|-----------|
| DP | 0.0 | 0.152 | 0.146 | 3/3 | +0.002 |
| DP | 0.1 | 0.207 | 0.205 | 2/3 | +0.001 |
| DP | 0.2 | 0.248 | 0.237 | 3/3 | +0.002 |
| DP | 0.3 | 0.286 | 0.264 | 3/3 | +0.009 |
| DP | 0.4 | 0.310 | 0.283 | 3/3 | +0.011 |
| Combined | 0.2 | 0.199 | 0.183 | 3/3 | +0.005 |
| Combined | 0.3 | 0.219 | 0.195 | 3/3 | +0.008 |
| Combined | 0.4 | 0.213 | 0.185 | 3/3 | +0.015 |

Source: `results/tau_ablation_tau1.json` (Adult complete, Credit/LSAC in progress)

## Why the previous story was wrong

At tau=100 (stepped schedule: tau=100 for alpha<=0.3, tau=1 for alpha=0.4):
- alpha=0.2: Naive 0.327 vs DRO 0.503 (DRO loses badly)
- alpha=0.3: Naive 0.531 vs DRO 0.562 (DRO loses)

At tau=1 (fixed):
- alpha=0.2: Naive 0.248 vs DRO 0.237 (DRO wins)
- alpha=0.3: Naive 0.286 vs DRO 0.264 (DRO wins)

Source: `results/tau_ablation_tau{1,10,100}.json`

## IF k-NN ablation

k in {5,10,15} give near-identical IF and DP values (+/-0.003).
The IF attack is robust to graph choice; k=5 is safe.

Source: `results/knn_ablation_k{5,10,15}.json` (Adult only)

## Adversarial vs random corruption

At matched alpha, adversarial PGD raises DP 12-40x more than random noise:
- Adult alpha=0.2: adv +0.18 vs random +0.001
- Adult alpha=0.3: adv +0.38 vs random +0.02

Source: `results/random_vs_adversarial_new.json`

## Statistical significance caveat

n=3 seeds cannot achieve p<0.05 (minimum achievable Wilcoxon p is 0.125).
n=6 re-run in progress. Per-seed win counts reported in lieu.

## Status

- Adult tau=1: COMPLETE (90 configs, 3 seeds)
- Credit/LSAC tau=1: IN PROGRESS (270-row target)
- n=6 seeds: IN PROGRESS
- UTKFace: BLOCKED on flair2 GPU access
