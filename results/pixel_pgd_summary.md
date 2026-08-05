# UTKFace pixel-space PGD (U3) — PARTIAL

rows: **4/24**

Protocol: train on **pixel** PGD (ε=4/255, steps=10) over raw UTKFace JPEGs; eval clean test DP/IF/acc. τ=1, k_inner=10, epochs=60. data_provenance=REAL_PIXELS.

wins = DRO strict lower DP (ties if |N−D| < 1e-5).

| α | n | DP N | DP D | wins (D/tie/n) | mean ΔDP (N−D) | acc N | acc D | IF N | IF D |
|---:|--:|-----:|-----:|---------------:|---------------:|------:|------:|-----:|-----:|
| 0.1 | 2 | 0.0157 | 0.0131 | 1/0/2 | +0.0026 | 0.8578 | 0.8595 | 0.0743 | 0.0594 |
| 0.2 | 2 | 0.0157 | 0.0195 | 0/0/2 | -0.0038 | 0.8547 | 0.8605 | 0.0725 | 0.0569 |

### Per-seed @ α=0.2 (n=2)

- s0: DP N=0.0155 D=0.0193 → **Naive** (acc N=0.856 D=0.864; t=211s)
- s1: DP N=0.0159 D=0.0197 → **Naive** (acc N=0.853 D=0.857; t=202s)

### Contrast to U1 feature-space (clean test DP means)

Not apples-to-apples: U1 corrupts **cached 512-d features** via FairnessTargetedPGD; U3 corrupts **pixels** then re-extracts features. Both report clean-test DP after train-time attack.

| α | U3 n | U3 DP N/D | U1-dp n | U1 clean DP N/D | U1 corr DP N/D |
|---:|-----:|----------:|--------:|----------------:|---------------:|
| 0.1 | 2 | 0.0157/0.0131 | 6 | 0.0480/0.0519 | 0.1788/0.1772 |
| 0.2 | 2 | 0.0157/0.0195 | 6 | 0.1572/0.1590 | 0.3342/0.3295 |

device=cuda flair2. target 24 cells (6 seeds × α∈{0.1,0.2}).
**PARTIAL** — not for paper claims until 24/24.
