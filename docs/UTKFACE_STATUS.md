# UTKFace Status (LIVE)

**Updated:** 2026-08-04 ~15:20  
**Path:** **2 — REAL local MPS** (Kaggle features). **No paper claim yet.**

---

## flair2 GPU — PROVEN & STAGED, PARKED

| Item | Status |
|------|--------|
| Access / SSH | ✅ |
| Driver (Supin) | ✅ 2× L40S, driver 570 |
| Code + real features on server | ✅ staged under `/data/srujan.sai/` |
| torch CUDA install | ⏸ **stopped** (campus wifi ~178 kB/s; wheelhouse saved on Mac) |
| GPU experiment launch | **Do not wait. Do not restart** unless pixel-PGD / GPU-heavy work is explicitly greenlit |

Mac already produces **REAL** UTKFace rows. flair2 is infrastructure for later, not a dependency.

---

## Local REAL grid (this machine)

| Item | Value |
|------|-------|
| Features | `data/raw/utkface_features.npz` — **23,705** × 512, provenance REAL |
| Output | `results/utkface_canonical.json` |
| Protocol | τ=1, k_inner=10, epochs=60, pgd_steps=20, 6 seeds, α∈{0,0.1,0.2,0.3,0.4} |
| Runner | **one** `experiments/run_utkface_server.py` (resume-safe; duplicates killed 15:20) |
| Log | `logs/utkface_canonical_run.log` |

### Progress (recompute anytime)

```bash
python3 -c "import json,collections;u=json.load(open('results/utkface_canonical.json'));print(len(u),dict(collections.Counter(r.get('attack') for r in u)))"
```

| Attack | Target | Status (last check ~51 rows) |
|--------|--------|------------------------------|
| **dp** | 30 | **DONE** (5 α × 6 seeds) |
| **if** | 30 | **in progress** (~21/30) |
| **combined** | 30 | **pending** |
| **Total** | **90** | ~51/90 REAL |

All rows: `data_provenance=REAL`. Synthetic fallback is off by default.

---

## Forbidden

- Do **not** report synthetic UTKFace as real  
- Do **not** block meeting or Aug 10 on flair2  
- Do **not** write `results/canonical_tau1.json` (tabular 540 frozen)

---

## When 90/90 completes

1. Commit `results/utkface_canonical.json`  
2. Summarize means / win counts (honest, including losses)  
3. Optional short paper/report subsection **or** formal scope-out if quality insufficient  
