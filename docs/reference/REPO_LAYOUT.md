# Repo layout (one screen)

```
DRO-FairML/
├── README.md              Project overview + reproduce path
├── STATUS.md              Live completion board (540 rows, IF MIXED, remaining work)
├── Makefile               install / test / validate / artifacts / paper / report
├── main.py                Legacy CLI: --generate-results (used by make results)
├── requirements.txt       Python deps
│
├── src/                   Implementation
│   ├── training/          dro_fair.py (Alg. 1), naive_fair.py
│   ├── corruption/        adversarial.py → FairnessTargetedPGD
│   ├── metrics/           DP, IF, accuracy
│   └── ...                models, radii, data utils
│
├── experiments/           Runners, analysis, plots (not the library)
│   ├── run_canonical.py   Canonical τ=1 / K=10 / n=6 → results/canonical_tau1.json
│   ├── validate_results.py / compute_canonical_wilcoxon.py
│   ├── generate_report_tables.py / generate_all_deliverables.py / generate_results.py
│   └── _archive/          Old one-off runners + legacy fig gens
│
├── results/               Committed experiment outputs (source of paper numbers)
│   ├── canonical_tau1.json          ★ 540-row grid
│   ├── canonical_wilcoxon.{csv,md}
│   ├── if_wilcoxon_summary.txt
│   └── stale_archived/              Old partial JSON (do not use for claims)
│
├── figures/               PDF/PNG figures for paper & meeting
├── paper/                 main.tex → main.pdf (+ auto_generated/*.tex)
├── report/                report.tex → report.pdf (+ sections/auto_generated_*.tex)
│
├── docs/                  Design, meeting, verification
│   ├── MEETING_2026-08-04.md   Meeting brief (honest tables)
│   ├── CLEAN_TREE.md           Post-cleanup “what lives where”
│   ├── REPO_LAYOUT.md          ← you are here
│   ├── INDEX.md                Doc index
│   ├── VERIFICATION_REPORT.md
│   ├── LSAC_DEGENERACY.md
│   └── _archive/               Superseded notes, old prompts, strays
│
├── tests/                 pytest suite
├── data/                  download_data.sh + raw tabular files
├── configs/               default.yaml
├── scripts/               5 ops helpers (finalize / UTKFace / flair2); rest in _archive/
└── logs/                  Run logs; archive_root/ holds moved root noise
```

## Where to look

| Goal | Go to |
|------|--------|
| What works / what's left | `STATUS.md` |
| Install & reproduce | `README.md` |
| Meeting numbers | `docs/MEETING_2026-08-04.md` |
| Train algorithm | `src/training/dro_fair.py` |
| Attack | `src/corruption/adversarial.py` |
| Canonical numbers | `results/canonical_tau1.json` |
| Rebuild PDFs only | `make paper` / `make report` |
| Full artifact regen (no train) | `make full` |

## Do not

- Re-run 540 configs just to open the paper — JSON is committed.
- Cite `results/stale_archived/` or τ=100 ablations as the main claim.
- Claim a clean IF three-attack sweep (IF is **MIXED**).
