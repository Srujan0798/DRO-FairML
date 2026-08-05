# UTKFace pixel-space PGD (U3) — PARTIAL

rows: **7/24**

Protocol: train on **pixel** PGD (ε=4/255, steps=10) over raw UTKFace JPEGs; eval clean test DP/IF/acc. τ=1, k_inner=10, epochs=60. data_provenance=REAL_PIXELS.

wins = DRO strict lower DP (ties if |N−D| < 1e-5).

| α | n | DP N | DP D | wins_DP (D/tie/n) | mean ΔDP | wins_IF (D/tie/n) | IF N | IF D | acc N | acc D |
|---:|--:|-----:|-----:|-----------------:|---------:|-----------------:|-----:|-----:|------:|------:|
| 0.1 | 4 | 0.0208 | 0.0170 | 3/0/4 | +0.0037 | 4/0/4 | 0.0774 | 0.0631 | 0.8604 | 0.8608 |
| 0.2 | 3 | 0.0182 | 0.0204 | 1/0/3 | -0.0022 | 3/0/3 | 0.0744 | 0.0589 | 0.8582 | 0.8626 |

### Per-seed @ α=0.2 (n=3)

- s0: DP N=0.0155 D=0.0193 → **Naive** (acc N=0.856 D=0.864; t=211s)
- s1: DP N=0.0159 D=0.0197 → **Naive** (acc N=0.853 D=0.857; t=202s)
- s2: DP N=0.0232 D=0.0223 → **DRO** (acc N=0.865 D=0.867; t=195s)

### Contrast to U1 feature-space (clean test DP means)

Not apples-to-apples: U1 corrupts **cached 512-d features** via FairnessTargetedPGD; U3 corrupts **pixels** then re-extracts features. Both report clean-test DP after train-time attack.

| α | U3 n | U3 DP N/D | U1-dp n | U1 clean DP N/D | U1 corr DP N/D |
|---:|-----:|----------:|--------:|----------------:|---------------:|
| 0.1 | 4 | 0.0208/0.0170 | 6 | 0.0480/0.0519 | 0.1788/0.1772 |
| 0.2 | 3 | 0.0182/0.0204 | 6 | 0.1572/0.1590 | 0.3342/0.3295 |

device=cuda flair2. target 24 cells (6 seeds × α∈{0.1,0.2}).
**PARTIAL** — not for paper claims until 24/24.
