# UTKFace multi-group (5-race) summary — PARTIAL

rows: **20/30**

| α | n | DP_bin N | DP_bin D | wins_bin | DP_multi N | DP_multi D | wins_multi | mean Δmulti (N−D) |
|---:|--:|---------:|---------:|---------:|-----------:|-----------:|-----------:|------------------:|
| 0.0 | 6 | 0.0211 | 0.0204 | 3/6 | 0.1282 | 0.1209 | 6/6 | +0.0073 |
| 0.1 | 6 | 0.0480 | 0.0519 | 0/6 | 0.1319 | 0.1283 | 4/6 | +0.0037 |
| 0.2 | 6 | 0.1572 | 0.1590 | 2/6 | 0.2256 | 0.2218 | 5/6 | +0.0038 |
| 0.3 | 2 | 0.1874 | 0.1806 | 2/2 | 0.2417 | 0.2337 | 2/2 | +0.0080 |

### DRO group positive rates (mean over seeds)

- α=0.0 n=6: max **Other** 0.565 / min **Black** 0.448 — {'White': 0.466, 'Black': 0.448, 'Asian': 0.538, 'Indian': 0.453, 'Other': 0.565}
- α=0.1 n=6: max **Other** 0.544 / min **White** 0.416 — {'White': 0.416, 'Black': 0.433, 'Asian': 0.514, 'Indian': 0.437, 'Other': 0.544}
- α=0.2 n=6: max **Other** 0.510 / min **White** 0.288 — {'White': 0.288, 'Black': 0.418, 'Asian': 0.484, 'Indian': 0.423, 'Other': 0.51}
- α=0.3 n=2: max **Other** 0.532 / min **White** 0.298 — {'White': 0.298, 'Black': 0.459, 'Asian': 0.513, 'Indian': 0.45, 'Other': 0.532}

Protocol: train DP on binary race (White vs non-White); eval max-min DP on 5 race groups.
REAL ResNet18 features. device=cuda flair2.
**PARTIAL** — not for paper claims until 30/30.
