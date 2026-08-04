# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~01:49 IST_

## Counts (do not pkill these jobs)

| Job | File (flair2 `DRO-FairML-run/results/`) | Target | Count | Process | GPU |
|-----|------------------------------------------|--------|-------|---------|-----|
| **U1** CUDA repro | `utkface_flair2.json` | **90** | **12** | PID 3482009 `run_utkface_server.py` **alive** (~1.8 h) | GPU0 ~637 MiB |
| **U2** 5-race | `utkface_multigroup.json` | **30** | **11** | PID 3482442 `run_utkface_multigroup.py` **alive** (~1.7 h) | GPU1 ~637 MiB |

- CWD both: `/data/srujan.sai/DRO-FairML-run`
- Mac puller: `scripts/u12_puller.sh` → `logs/u12_puller.log` (independent finalize; 10-min partial rsync)
- GPU util often ~0% (CPU-heavy feature path); ~15–18 min / attacked seed

### Position
- **U1:** `dp` **α=0.0 and α=0.1 complete (12/12)**. Next: dp α=0.2 seed=0. Then α=0.3–0.4, then if, then combined.
- **U2:** α=0.0 = 6/6; α=0.1 = 5/6 (seed 5 mid-run since ~01:33).

### Pace / ETA
- U1 remaining 78 → **~22 h**
- U2 remaining 19 → **~5.4 h**

## Early signals (honest, partial)

### U1 CUDA vs Mac MPS — checkpoint α≤0.1
Seed-wise table: `results/utkface_repro_checkpoint_a01.md` (**12** matched cells).
- max \|ΔDP_dro\| = **0.0072**, max \|Δacc_dro\| = **0.0025**, **0 GAP** cells (thr 0.02)
- Mean ΔDP_dro ≈ +0.0008. Strong intermediate CUDA↔MPS agreement for completed dp strip.
- Still **not** a full repro claim until if/combined + higher α finish.

### U2 multi-group
| α | n | DP_bin N/D | wins_bin | DP_multi N/D | wins_multi |
|---:|--:|------------|----------|--------------|------------|
| 0.0 | 6 | 0.021/0.020 | 3/6 | 0.128/0.121 | **6/6** |
| 0.1 | 5 | 0.048/0.052 | 0/5 | 0.132/0.127 | **4/5** |

**Group extremum (DRO):** max-min gap driven by **Other** (highest pos rate ~0.54–0.56) vs Black/Indian/White (lowest). Asian also elevated (~0.51–0.54). Binary DP under attack currently favors Naive (0/5 at α=0.1) — do not headline until full grid.

## U3 pixel PGD
**Blocked:** no raw UTKFace JPEGs on flair2. Features-only. `src/corruption/image_pgd.py` present for later.

## Tick history (brief)
- **01:30:** U1=11 U2=10; device provenance + flush; `ff7c430`
- **01:40:** U1=11 U2=11; independent puller; `7985eda`
- **01:49:** U1=12 U2=11; dp α≤0.1 complete + seed-wise repro checkpoint (0 GAP)

## Open
- Wait U1=90 / U2=30 (U2 first ~morning)
- U3 needs JPEG tree
- Finding 3 cosine disclosure → paper integration pass
