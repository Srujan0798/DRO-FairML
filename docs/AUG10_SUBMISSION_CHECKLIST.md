# Aug 10 submission checklist

## Share externally
- `docs/MEETING_HANDOUT_2026-08-04.md` (already used for meeting)
- `paper/main.pdf` + `report/report.pdf` (after rebuild)
- Optional: `results/utkface_summary.md` if asked about images

## Locked science
- Tabular: `results/canonical_tau1.json` — **540 rows**
- UTKFace: `results/utkface_canonical.json` — **90/90 REAL**
- IF Wilcoxon: `results/if_wilcoxon_summary.txt`

## Claims (must match)
1. τ=1 makes DRO robust on Adult & Credit at α≤0.2 (DP+Combined)
2. Adult/DP α=0.1 is **5/6**, not 6/6 every α
3. IF attack **mixed**
4. LSAC/DP **degenerate**
5. α≥0.3: below constant predictor on Adult/Credit only
6. UTKFace: real pilot; **mixed clean-test**; not Adult copy-paste

## Build
```bash
make test && make validate && make paper && make report
```

## Do not
- Retrain 540
- Claim synthetic UTKFace
- Claim clean UTKFace low-α sweep without looking at utkface_summary.md
