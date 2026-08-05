# UTKFace pixel-space PGD (U3) — PARTIAL

rows: **2/24**

Protocol: train on **pixel** PGD (ε=4/255, steps=10) over raw UTKFace JPEGs; eval clean test DP/IF/acc. τ=1, k_inner=10, epochs=60. data_provenance=REAL_PIXELS.

wins = DRO strict lower DP (ties if |N−D| < 1e-5).

| α | n | DP N | DP D | wins (D/tie/n) | mean ΔDP (N−D) | acc N | acc D | IF N | IF D |
|---:|--:|-----:|-----:|---------------:|---------------:|------:|------:|-----:|-----:|
| 0.1 | 1 | 0.0151 | 0.0097 | 1/0/1 | +0.0054 | 0.8576 | 0.8616 | 0.0734 | 0.0626 |
| 0.2 | 1 | 0.0155 | 0.0193 | 0/0/1 | -0.0038 | 0.8561 | 0.8635 | 0.0701 | 0.0559 |

### Per-seed @ α=0.2 (n=1)

- s0: DP N=0.0155 D=0.0193 → **Naive** (acc N=0.856 D=0.864; t=211s)

### Contrast to U1 feature-space (clean test DP means)

Not apples-to-apples: U1 corrupts **cached 512-d features** via FairnessTargetedPGD; U3 corrupts **pixels** then re-extracts features. Both report clean-test DP after train-time attack.

| α | U3 n | U3 DP N/D | U1-dp n | U1 clean DP N/D | U1 corr DP N/D |
|---:|-----:|----------:|--------:|----------------:|---------------:|
| 0.1 | 1 | 0.0151/0.0097 | 6 | 0.0480/0.0519 | 0.1788/0.1772 |
| 0.2 | 1 | 0.0155/0.0193 | 6 | 0.1572/0.1590 | 0.3342/0.3295 |

device=cuda flair2. target 24 cells (6 seeds × α∈{0.1,0.2}).
**PARTIAL** — not for paper claims until 24/24.
