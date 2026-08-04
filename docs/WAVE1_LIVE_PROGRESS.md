# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~02:30 IST_

## Counts (do not pkill)

| Job | Target | Count | Notes |
|-----|--------|------:|-------|
| **U1** | 90 | **14** (live may be same mid-cell) | PID 3482009 GPU0; last written dp α=0.2 seed=1; **actively computing** seed=2 |
| **U2** | 30 | **14** | PID 3482442 GPU1; mid α=0.2 seed=2 |

Puller OK. Jobs confirmed R-state with rising CPU ticks (not hung).

## Signals
- Repro summarizer now reports **clean + corrupted** DP/acc deltas.
- Local mirror: max\|ΔDP clean\|=0.0072, max\|ΔDP corrupted\|=0.0064 (0 GAP).
- U3: ImageNet ResNet18 weights **cached** on flair2 (`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`).

## Finding 3
- Paper now states cosine IF ≠ original Euclidean IF (`paper/sections/results.tex`).

## Open
- Wait U1=90 / U2=30; U3 after free GPU + CONFIRM symlink.
