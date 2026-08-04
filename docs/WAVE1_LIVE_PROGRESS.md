# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~01:40 IST_

## Counts (do not pkill these jobs)

| Job | File (flair2 `DRO-FairML-run/results/`) | Target | Count | Process | GPU |
|-----|------------------------------------------|--------|-------|---------|-----|
| **U1** CUDA repro | `utkface_flair2.json` | **90** | **11** | PID 3482009 `run_utkface_server.py` **alive** | `CUDA_VISIBLE_DEVICES=0` ~637 MiB |
| **U2** 5-race | `utkface_multigroup.json` | **30** | **11** | PID 3482442 `run_utkface_multigroup.py` **alive** | `CUDA_VISIBLE_DEVICES=1` ~637 MiB |

- CWD both: `/data/srujan.sai/DRO-FairML-run`
- Mac puller: `scripts/u12_puller.sh` → `logs/u12_puller.log` (**independent** finalize when U1=90 or U2=30; partial rsync every 10 min)
- GPU util often ~0% with high CPU: feature-space path is CPU-heavy; rows still advance (~15–18 min / attacked seed)

### Position
- **U1:** `dp` α=0.0 = 6/6; α=0.1 = 5/6 (seeds 0–4). Last written seed=4 @ 01:29; mid-cell for seed 5 (normal ~18 min).
- **U2:** α=0.0 = 6/6; α=0.1 = 5/6 (seeds 0–4 done); log: `RUN a=0.1 s=5` in progress.

### Pace / ETA (attacked-row `total_time` ≈ 1000 s)
- U1 remaining ~79 → **~22 h**
- U2 remaining ~19 → **~5.3 h**

## Early signals (honest, partial)

### U1 CUDA vs Mac MPS
Matched **11** cells (dp α∈{0.0,0.1}): max \|Δ DP_dro\|≈**0.007**, max \|Δ acc_dro\|≈**0.0025**. No GAP. Not a full repro claim until 90/90.
U1 rows lack top-level `device` (job predated provenance fix); args are `--device cuda`.

### U2 multi-group
| α | n | DP_bin N/D | wins_bin | DP_multi N/D | wins_multi |
|---:|--:|------------|----------|--------------|------------|
| 0.0 | 6 | 0.021/0.020 | 3/6 | 0.128/0.121 | **6/6** |
| 0.1 | 5 | 0.048/0.052 | 0/5 | 0.132/0.127 | **4/5** |

**Group extremum (DRO):** max-min gap driven by **Other** (highest pos rate ~0.54–0.56) vs Black/Indian/White (lowest). Asian also elevated (~0.51–0.54). Binary DP under attack currently favors Naive (0/5 at α=0.1) — do not headline until full grid.

## U3 pixel PGD
**Blocked:** no raw UTKFace JPEGs on flair2. Features-only. `src/corruption/image_pgd.py` present for later.

## Tick history (brief)
- **01:30:** U1=11 U2=10; progress doc; device provenance + flush prints; partial rsync; commit `ff7c430`
- **01:40:** U1=11 U2=11; durable `scripts/u12_puller.sh` (independent finalize + 10-min partial rsync); group-rate note; interim multigroup summary

## Open
- Wait U1=90 / U2=30 (U2 first ~tonight)
- U3 needs JPEG tree
- Finding 3 cosine disclosure → paper integration pass
