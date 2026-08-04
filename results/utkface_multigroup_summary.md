# UTKFace multi-group (5-race) summary — PARTIAL

rows: **23/30**

wins = DRO strict lower DP; ties if |N−D| < 1e-5 (not counted as wins).

| α | n | DP_bin N | DP_bin D | wins_bin (D/tie/n) | DP_multi N | DP_multi D | wins_multi (D/tie/n) | mean Δmulti (N−D) |
|---:|--:|---------:|---------:|------------------:|-----------:|-----------:|---------------------:|------------------:|
| 0.0 | 6 | 0.0211 | 0.0204 | 3/0/6 | 0.1282 | 0.1209 | 6/0/6 | +0.0073 |
| 0.1 | 6 | 0.0480 | 0.0519 | 0/0/6 | 0.1319 | 0.1283 | 4/0/6 | +0.0037 |
| 0.2 | 6 | 0.1572 | 0.1590 | 2/0/6 | 0.2256 | 0.2218 | 5/0/6 | +0.0038 |
| 0.3 | 5 | 0.1834 | 0.1814 | 3/0/5 | 0.2405 | 0.2356 | 4/1/5 | +0.0050 |

### DRO group positive rates (mean over seeds)

- α=0.0 n=6: max **Other** 0.565 / min **Black** 0.448 — {'White': 0.466, 'Black': 0.448, 'Asian': 0.538, 'Indian': 0.453, 'Other': 0.565}
- α=0.1 n=6: max **Other** 0.544 / min **White** 0.416 — {'White': 0.416, 'Black': 0.433, 'Asian': 0.514, 'Indian': 0.437, 'Other': 0.544}
- α=0.2 n=6: max **Other** 0.510 / min **White** 0.288 — {'White': 0.288, 'Black': 0.418, 'Asian': 0.484, 'Indian': 0.423, 'Other': 0.51}
- α=0.3 n=5: max **Other** 0.535 / min **White** 0.300 — {'White': 0.3, 'Black': 0.457, 'Asian': 0.522, 'Indian': 0.452, 'Other': 0.535}

### Per-seed multi @ α=0.3 (n=5)

- s0: multi N=0.2366 D=0.2338 → **DRO** (bin N=0.1804 D=0.1788 → DRO)
- s1: multi N=0.2468 D=0.2336 → **DRO** (bin N=0.1943 D=0.1825 → DRO)
- s2: multi N=0.2573 D=0.2493 → **DRO** (bin N=0.1822 D=0.1780 → DRO)
- s3: multi N=0.2341 D=0.2341 → **tie** (bin N=0.1887 D=0.1919 → Naive)
- s4: multi N=0.2279 D=0.2271 → **DRO** (bin N=0.1712 D=0.1759 → Naive)

Protocol: train DP on binary race (White vs non-White); eval max-min DP on 5 race groups.
REAL ResNet18 features. device=cuda flair2.
**PARTIAL** — not for paper claims until 30/30.
