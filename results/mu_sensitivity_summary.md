# TASK C — mu sensitivity curve (pre-registered)

6-point curve: mu in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
Datasets: ['adult', 'credit'] | alpha=0.2 | 6 seeds | methods: naive + dro
Data: mu_sensitivity.json (144 rows) + aug_lagrangian.json (48 rows, mu=5,10 reused)

## DP violation vs mu (alpha=0.2, attack=dp)

| mu | Adult naive | Adult DRO | Credit naive | Credit DRO |
|---|---|---|---|---|
| 0.5 | 0.2438 | 0.2164 | 0.0197 | 0.0129 |
| 1 | 0.2438 | 0.2025 | 0.0197 | 0.0090 |
| 2 | 0.2438 | 0.1792 | 0.0197 | 0.0057 |
| 5 | 0.2465 | 0.1358 | 0.0206 | 0.0026 |
| 10 | 0.2465 | 0.1009 | 0.0206 | 0.0017 |
| 20 | 0.2438 | 0.0682 | 0.0197 | 0.0013 |

## Accuracy vs mu (alpha=0.2, attack=dp)

| mu | Adult floor+0.02 | Adult naive | Adult DRO | Credit floor+0.02 | Credit naive | Credit DRO |
|---|---|---|---|---|---|---|
| 0.5 | 0.7721 | 0.7570 | 0.7694 | 0.7988 | 0.7831 | 0.7808 |
| 1 | 0.7721 | 0.7570 | 0.7698 | 0.7988 | 0.7831 | 0.7780 |
| 2 | 0.7721 | 0.7570 | 0.7732 | 0.7988 | 0.7831 | 0.7784 |
| 5 | 0.7721 | 0.7595 | 0.7944 | 0.7988 | 0.7820 | 0.7730 |
| 10 | 0.7721 | 0.7595 | 0.7953 | 0.7988 | 0.7820 | 0.7699 |
| 20 | 0.7721 | 0.7570 | 0.7783 | 0.7988 | 0.7831 | 0.7684 |

## Degeneracy threshold and recommended mu

Rule C2: recommended mu = **largest mu whose mean DRO accuracy stays >= floor + 0.02**. Any DP improvement at/below the floor is DEGEN (collapse, not fairness).

| dataset | floor | floor+0.02 | threshold mu* | DRO acc at mu* | verdict |
|---|---|---|---|---|---|
| adult | 0.7521 | 0.7721 | 20 | 0.7783 | threshold: collapse below floor at not reached in [0.5, 20] |
| credit | 0.7788 | 0.7988 | 0.5 | 0.7808 | **no mu reaches floor+0.02; AL unsafe** |

## Per-mu degeneracy flags (DRO arm)

Any (mu, dataset) where DRO mean accuracy <= floor is flagged **DEGEN**: the apparent DP improvement is constant-predictor collapse, not fairness.

| mu | adult DRO acc | adult dp | adult floor | adult verdict | credit DRO acc | credit dp | credit floor | credit verdict |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.7694 | 0.2164 | 0.7521 | ok | 0.7808 | 0.0129 | 0.7788 | ok |
| 1 | 0.7698 | 0.2025 | 0.7521 | ok | 0.7780 | 0.0090 | 0.7788 | **DEGEN** |
| 2 | 0.7732 | 0.1792 | 0.7521 | ok | 0.7784 | 0.0057 | 0.7788 | **DEGEN** |
| 5 | 0.7944 | 0.1358 | 0.7521 | ok | 0.7730 | 0.0026 | 0.7788 | **DEGEN** |
| 10 | 0.7953 | 0.1009 | 0.7521 | ok | 0.7699 | 0.0017 | 0.7788 | **DEGEN** |
| 20 | 0.7783 | 0.0682 | 0.7521 | ok | 0.7684 | 0.0013 | 0.7788 | **DEGEN** |

## Recommendation

- **adult**: mu=20 (largest tested mu with mean accuracy 0.7783 >= floor+0.02=0.7721).
- **credit**: **no safe mu** — even mu=0.5 gives mean accuracy 0.7808 < floor+0.02=0.7988. Do not use AL on credit.

Degeneracy guard enforced: any DP improvement at/below the floor (adult=0.7521, credit=0.7788) is labelled **DEGEN**, not a win.
