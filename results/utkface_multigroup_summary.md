# UTKFace multi-group (5-race) summary — PARTIAL

rows: **12/30** (live flair2 CUDA)

| α | n | DP_bin N | DP_bin D | wins_bin | DP_multi N | DP_multi D | wins_multi | mean Δmulti (N−D) |
|---:|--:|---------:|---------:|---------:|-----------:|-----------:|-----------:|------------------:|
| 0.0 | 6 | 0.0211 | 0.0204 | 3/6 | 0.1282 | 0.1209 | 6/6 | +0.0073 |
| 0.1 | 6 | 0.0480 | 0.0519 | 0/6 | 0.1319 | 0.1283 | 4/6 | +0.0037 |

### Seed detail α=0.1 (complete 6/6)

| seed | bin N | bin D | multi N | multi D | multi win? | Other_pos dro | White_pos dro |
|-----:|------:|------:|--------:|--------:|:-----------:|-------------:|--------------:|
| 0 | 0.0459 | 0.0481 | 0.1171 | 0.1172 | N | 0.527 | 0.410 |
| 1 | 0.0506 | 0.0564 | 0.1328 | 0.1320 | Y | 0.554 | 0.422 |
| 2 | 0.0443 | 0.0449 | 0.1416 | 0.1356 | Y | 0.553 | 0.418 |
| 3 | 0.0567 | 0.0618 | 0.1308 | 0.1275 | Y | 0.537 | 0.409 |
| 4 | 0.0420 | 0.0469 | 0.1353 | 0.1228 | Y | 0.538 | 0.420 |
| 5 | 0.0485 | 0.0533 | 0.1340 | 0.1344 | N | 0.552 | 0.418 |

### DRO mean group rates by α

- α=0.0 n=6: max **Other** 0.565 / min **Black** 0.448 — {'White': 0.466, 'Black': 0.448, 'Asian': 0.538, 'Indian': 0.453, 'Other': 0.565}
- α=0.1 n=6: max **Other** 0.544 / min **White** 0.416 — {'White': 0.416, 'Black': 0.433, 'Asian': 0.514, 'Indian': 0.437, 'Other': 0.544}

### Early read (not paper-ready)
- α=0.0 multi: DRO wins 6/6 on max-min DP; binary ~coin flip.
- α=0.1 multi: DRO wins **4/6**; binary **0/6** (DRO often slightly worse on binary).
- Max-min gap remains **Other-high vs White/Black/Indian-low** under attack.
- **PARTIAL** until 30/30. Protocol: train binary race DP; eval 5-way max-min. REAL features, cuda flair2.
