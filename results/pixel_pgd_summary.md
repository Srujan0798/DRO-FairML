# UTKFace pixel-space PGD (U3) — PARTIAL

rows: **11/24**

Protocol: train on **pixel** PGD (ε=4/255, steps=10) over raw UTKFace JPEGs; eval clean test DP/IF/acc. τ=1, k_inner=10, epochs=60. data_provenance=REAL_PIXELS.

wins = DRO strict lower DP (ties if |N−D| < 1e-5).

| α | n | DP N | DP D | wins_DP (D/tie/n) | mean ΔDP | wins_IF (D/tie/n) | IF N | IF D | acc N | acc D |
|---:|--:|-----:|-----:|-----------------:|---------:|-----------------:|-----:|-----:|------:|------:|
| 0.1 | 6 | 0.0243 | 0.0224 | 4/0/6 | +0.0019 | 6/0/6 | 0.0764 | 0.0627 | 0.8593 | 0.8601 |
| 0.2 | 5 | 0.0228 | 0.0241 | 2/0/5 | -0.0014 | 5/0/5 | 0.0747 | 0.0600 | 0.8581 | 0.8607 |

### Per-seed @ α=0.2 (n=5)

- s0: DP N=0.0155 D=0.0193 → **Naive** (acc N=0.856 D=0.864; t=211s)
- s1: DP N=0.0159 D=0.0197 → **Naive** (acc N=0.853 D=0.857; t=202s)
- s2: DP N=0.0232 D=0.0223 → **DRO** (acc N=0.865 D=0.867; t=195s)
- s3: DP N=0.0285 D=0.0237 → **DRO** (acc N=0.857 D=0.856; t=203s)
- s4: DP N=0.0309 D=0.0358 → **Naive** (acc N=0.859 D=0.860; t=201s)

### Contrast to U1 feature-space (clean test DP means)

Not apples-to-apples: U1 corrupts **cached 512-d features** via FairnessTargetedPGD; U3 corrupts **pixels** then re-extracts features. Both report clean-test DP after train-time attack.

| α | U3 n | U3 DP N/D | U1-dp n | U1 clean DP N/D | U1 corr DP N/D |
|---:|-----:|----------:|--------:|----------------:|---------------:|
| 0.1 | 6 | 0.0243/0.0224 | 6 | 0.0480/0.0519 | 0.1788/0.1772 |
| 0.2 | 5 | 0.0228/0.0241 | 6 | 0.1572/0.1590 | 0.3342/0.3295 |

device=cuda flair2. target 24 cells (6 seeds × α∈{0.1,0.2}).
**PARTIAL** — not for paper claims until 24/24.
