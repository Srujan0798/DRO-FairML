# UTKFace multi-group (5-race) summary — PARTIAL

rows: **27/30**

wins = DRO strict lower DP; ties if |N−D| < 1e-5 (not counted as wins).

| α | n | DP_bin N | DP_bin D | wins_bin (D/tie/n) | DP_multi N | DP_multi D | wins_multi (D/tie/n) | mean Δmulti (N−D) |
|---:|--:|---------:|---------:|------------------:|-----------:|-----------:|---------------------:|------------------:|
| 0.0 | 6 | 0.0211 | 0.0204 | 3/0/6 | 0.1282 | 0.1209 | 6/0/6 | +0.0073 |
| 0.1 | 6 | 0.0480 | 0.0519 | 0/0/6 | 0.1319 | 0.1283 | 4/0/6 | +0.0037 |
| 0.2 | 6 | 0.1572 | 0.1590 | 2/0/6 | 0.2256 | 0.2218 | 5/0/6 | +0.0038 |
| 0.3 | 6 | 0.1832 | 0.1808 | 4/0/6 | 0.2417 | 0.2368 | 5/1/6 | +0.0049 |
| 0.4 | 3 | 0.2264 | 0.2152 | 3/0/3 | 0.2774 | 0.2603 | 3/0/3 | +0.0170 |

### DRO group positive rates (mean over seeds)

- α=0.0 n=6: max **Other** 0.565 / min **Black** 0.448 — {'White': 0.466, 'Black': 0.448, 'Asian': 0.538, 'Indian': 0.453, 'Other': 0.565}
- α=0.1 n=6: max **Other** 0.544 / min **White** 0.416 — {'White': 0.416, 'Black': 0.433, 'Asian': 0.514, 'Indian': 0.437, 'Other': 0.544}
- α=0.2 n=6: max **Other** 0.510 / min **White** 0.288 — {'White': 0.288, 'Black': 0.418, 'Asian': 0.484, 'Indian': 0.423, 'Other': 0.51}
- α=0.3 n=6: max **Other** 0.537 / min **White** 0.301 — {'White': 0.301, 'Black': 0.457, 'Asian': 0.521, 'Indian': 0.452, 'Other': 0.537}
- α=0.4 n=3: max **Asian** 0.582 / min **White** 0.321 — {'White': 0.321, 'Black': 0.522, 'Asian': 0.582, 'Indian': 0.496, 'Other': 0.579}

### Per-seed multi @ α=0.4 (n=3)

- s0: multi N=0.2659 D=0.2558 → **DRO** (bin N=0.2206 D=0.2126 → DRO)
- s1: multi N=0.2814 D=0.2564 → **DRO** (bin N=0.2326 D=0.2180 → DRO)
- s2: multi N=0.2849 D=0.2688 → **DRO** (bin N=0.2262 D=0.2149 → DRO)
- bin↔multi winner agreement: **3/3** (disagreements=0; multi max-min can flip vs binary White/non-White)

Protocol: train DP on binary race (White vs non-White); eval max-min DP on 5 race groups.
REAL ResNet18 features. device=cuda flair2.
**PARTIAL** — not for paper claims until 30/30.
