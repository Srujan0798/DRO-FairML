# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:41 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **45** | if α=0.2 s2; dp done; if α=0.0–0.1 done | 3482009 | yes GPU0 |
| **U2** | 30 | **30** | **COMPLETE** | — | done |
| **U3** | 24 | **2** | s0 α=0.1 + α=0.2 (~220s/cell) | 3507049 | yes GPU1 |

Puller restarted with **U3 tracking** (won't exit at U1+U2 only). ETA U1 ~1.5 h; U3 ~1.5 h @ ~4 min/cell.

## Signals
- Repro: **45** matched (local after puller), **0 GAP**
- U3 s0: α=0.1 DRO wins DP (0.015→0.010); α=0.2 **Naive** (0.016→0.019) — partial only
- Pixel clean DP ≪ feature-space clean DP at same α (different attack; honest contrast table)

## This tick
- U3 first cells saved after OOM fix (stable)
- New `experiments/summarize_utkface_pixel_pgd.py` → `results/pixel_pgd_summary.md`
- Puller: U1+U2+U3 finalize; resume flags

## Open
- U1=90; U3=24; U2 done
