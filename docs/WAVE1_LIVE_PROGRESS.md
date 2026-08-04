# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~01:30 IST_

## Counts (do not pkill these jobs)

| Job | File (flair2 `DRO-FairML-run/results/`) | Target | Count | Process | GPU |
|-----|------------------------------------------|--------|-------|---------|-----|
| **U1** CUDA repro | `utkface_flair2.json` | **90** | **11** | PID 3482009 `run_utkface_server.py` (alive, ~01:26 elapsed) | `CUDA_VISIBLE_DEVICES=0` L40S ~628 MiB |
| **U2** 5-race | `utkface_multigroup.json` | **30** | **10** | PID 3482442 `run_utkface_multigroup.py` (alive, ~01:24 elapsed) | `CUDA_VISIBLE_DEVICES=1` L40S ~628 MiB |

- CWD both: `/data/srujan.sai/DRO-FairML-run`
- Mac puller: `logs/u12_puller.log` (2 min poll; rsync+summarize when 90+30)
- Partial Mac mirror: `results/utkface_flair2.json`, `results/utkface_multigroup.json` (this tick)
- GPU util often ~0% with high CPU: feature-space MLP + k-NN is CPU-heavy; rows still advancing (~15–18 min / attacked seed)

### Position
- **U1:** attack=`dp`, α=0.0 complete (6/6 seeds); α=0.1 seeds 0–4 written (last seed=4). Next: dp α=0.1 seed 5 → rest of dp → if → combined.
- **U2:** α=0.0 complete (6/6); α=0.1 seeds 0–3 written; log shows `RUN a=0.1 s=4` in progress.

### Pace / ETA (from attacked-row `total_time` ≈ 1000 s)
- U1 remaining ~79 cells → **~22 h** if pace holds
- U2 remaining ~20 cells → **~5.5 h**

## Early signals (honest, partial)

### U1 CUDA vs Mac MPS (`results/utkface_reproducibility_summary.md`)
Matched **11** cells so far (dp α∈{0.0,0.1}):
- max \|Δ DP_dro\| ≈ **0.007**, max \|Δ acc_dro\| ≈ **0.0025**
- mean Δ DP_dro ≈ +0.0008, mean Δ acc_dro ≈ 0
- No GAP flags yet. **Not a full repro claim** until 90/90.

Note: current U1 JSON rows lack a top-level `device` key (job started before provenance fix). Job args are `--device cuda` on GPU0; Mac summary treats them as flair2 CUDA. Future rows/resume will record `device` after Mac code sync.

### U2 multi-group (train binary race, eval max-min 5-way)
| α | n | DP_bin N/D | wins_bin | DP_multi N/D | wins_multi |
|---:|--:|------------|----------|--------------|------------|
| 0.0 | 6 | 0.021/0.020 | 3/6 | 0.128/0.121 | **6/6** |
| 0.1 | 4 | 0.049/0.053 | 0/4 | 0.131/0.128 | 3/4 |

Early read: multi-group gap favors DRO at α=0; under attack binary DP is mixed/against DRO so far — **do not headline until n=6 all α**.

## U3 pixel PGD
**Blocked:** no raw UTKFace JPEGs on flair2 (`/data/srujan.sai/UTKFace` missing). Only `utkface_features.npz` present. `src/corruption/image_pgd.py` is on the run tree for when images appear.

## This tick improvements
- Partial rsync of U1/U2 JSONs to Mac; ran `summarize_utkface_repro.py` (partial)
- `run_utkface.py`: write `device` into row provenance
- `run_utkface_server.py`: `flush=True` on prints (U1 live log is empty due to full buffering / no `PYTHONUNBUFFERED` — do **not** restart U1; fix applies on next launch)
- Did **not** rsync code over flair2 mid-run (live jobs keep loaded bytecode)

## Open
- Wait for U1=90, U2=30 → puller finalizes summaries + commit
- U3 needs JPEG tree on flair2
- Finding 3 cosine disclosure still for paper/report integration pass (not this lane)
