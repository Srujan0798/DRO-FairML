# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~23:05 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **90** | combined α=0.4 s5 | — | **COMPLETE** |
| **U2** | 30 | **30** | dp α=0.4 s5 | — | **COMPLETE** |
| **U3** | 12 | **12** | pixel_pgd α=0.2 s5 | — | **COMPLETE** |

GPUs idle (0 MiB). Images linked: `/data/srujan.sai/UTKFace` → kshitish (n_jpg=23708).

## Signals
- Repro U1: **90** matched, max|ΔDP_clean|=0.0072, max|ΔDP_corr|=0.0122 — all OK (`results/utkface_reproducibility_summary.md`)
- U2 multi-group: DRO multi wins 6/4/5/5/6 by α; 6/6 at α=0.4 (`results/utkface_multigroup_summary.md`)
- U3 pixel: intentional grid α∈{0.1,0.2} only (12 rows). DP mixed (4/6 @0.1, 2/6 @0.2); IF clean 6/6 both α (`results/pixel_pgd_summary.md`)

## This tick
- Fixed puller/status **U3 target 24→12** (matched HANDOFF + `run_utkface_pixel_pgd.py` defaults) so finalize can complete
- Fixed puller false-positive `alive|u1=1|u2=1|u3=1` (checker argv self-match → pgrep bracket patterns)
- Refreshed summaries; docs aligned with completion

## Open
- U1–U3 GPU lane **done**. U4 CelebA stretch optional only.
- Integration: paper/report already has cosine IF disclosure (Finding 3); leave further paper edits to integration pass
- Do not rewrite `canonical_tau1.json` / `utkface_canonical.json`
- **ATTENTION:** accidental `git checkout -- results/canonical_tau1.json` wiped uncommitted ~900-row expansion back to HEAD 568 rows — see `STATUS.md` Grok section; GLM may need to re-extend
