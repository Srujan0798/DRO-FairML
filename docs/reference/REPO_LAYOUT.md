# Repo layout (2026-08-04 — merged from CLEAN_TREE.md + REPO_LAYOUT.md)

Source of truth for tabular claims: `results/canonical_tau1.json` (**540** rows).
Source of truth for UTKFace claims: `results/utkface_canonical.json` (**REAL** rows only).

```
DRO-FairML/
├── README.md              Project overview + reproduce path
├── STATUS.md              Live completion board
├── Makefile               install / test / validate / artifacts / paper / report
├── main.py                Legacy CLI: --generate-results (used by make results)
├── requirements.txt setup.py LICENSE
│
├── src/                   Implementation
│   ├── training/          dro_fair.py (Alg. 1), naive_fair.py
│   ├── corruption/        adversarial.py → FairnessTargetedPGD
│   ├── temperature.py     get_temperature() — single source of truth, τ=1 fixed
│   └── ...                models, metrics, radii, data utils
│
├── experiments/           Runners, analysis, plots — ~19 files, all live/referenced
│   ├── run_canonical.py           Canonical τ=1 / K=10 / n=6 → results/canonical_tau1.json
│   ├── run_utkface_server.py      REAL UTKFace grid (GPU/MPS)
│   ├── validate_results.py / compute_canonical_wilcoxon.py
│   ├── generate_figures.py        Generates the 5 report-live figures (fig1/2/4/5/7)
│   ├── generate_report_tables.py / generate_all_deliverables.py / generate_results.py
│   └── (no _archive/ — superseded scripts live in git history, not the working tree)
│
├── results/               Committed experiment outputs (source of paper numbers)
│   ├── canonical_tau1.json          ★ 540-row tabular grid
│   ├── utkface_canonical.json       ★ REAL UTKFace grid (in progress)
│   ├── canonical_wilcoxon.{csv,md}, if_wilcoxon_summary.txt
│   └── table1_*, summary_stats.csv  (regen via make tables / make results)
│
├── figures/               PDF figures for paper & meeting (14 files, no PNGs)
├── paper/                 main.tex → main.pdf (+ auto_generated/*.tex)
├── report/                report.tex → report.pdf (+ sections/auto_generated_*.tex)
│
├── docs/
│   ├── INDEX.md                 Doc index
│   ├── STATUS.md-adjacent docs  MEETING_2026-08-04.md, KULDEEP_CORRECTION.md,
│   │                            LSAC_DEGENERACY.md, UTKFACE_STATUS.md, VERIFICATION_REPORT.md
│   └── reference/                Durable design docs + this file
│       ├── ARCHIVE_POLICY.md         The final policy (read this before deleting anything)
│       ├── ABLATION_STATUS_REPORT.md Cited by STATUS.md — load-bearing
│       ├── FAIRNESS_PGD_DESIGN.md, Q5_derivation.md, UTKFACE_PIPELINE.md, SERVER_RUNBOOK.md
│       └── TAU1_ABLATION_SUMMARY.md
│
├── tests/                 pytest suite (64 passing)
├── data/                  download_data.sh + raw tabular files (gitignored)
├── configs/               default.yaml
├── scripts/                4 operational helpers: agent_h_finalize.sh,
│                           deploy_utkface_flair2.sh, extract_utkface_features.py,
│                           flair2_ssh_config_snippet.txt
└── logs/                  Run logs (gitignored except README.md)
```

## Where to look

| Goal | Go to |
|------|--------|
| What works / what's left | `STATUS.md` |
| Install & reproduce | `README.md`, `make install && make data && make test` |
| Meeting numbers | `docs/MEETING_2026-08-04.md` |
| Train algorithm | `src/training/dro_fair.py` |
| Attack | `src/corruption/adversarial.py` |
| Canonical numbers | `results/canonical_tau1.json` |
| Rebuild PDFs only | `make paper` / `make report` |
| Full artifact regen (no train) | `make full` |
| Before deleting/archiving a file | `docs/reference/ARCHIVE_POLICY.md` |

## Makefile map → scripts

| Target | Entry |
|--------|-------|
| `test` | `pytest tests/` |
| `validate` | `experiments/validate_results.py` |
| `wilcoxon` | `experiments/compute_canonical_wilcoxon.py` |
| `tables` | `experiments/generate_report_tables.py` |
| `results` | `main.py --generate-results` → `generate_results.py` |
| `deliverables` | `experiments/generate_all_deliverables.py` |
| `paper` / `report` | tectonic |
| `full` | wilcoxon + tables + results + deliverables (no retrain) |

## Claims discipline

- IF story is **mixed**, not a clean three-attack sweep (see `docs/VERIFICATION_REPORT.md`)
- Adult/DP α=0.1 is **5/6** wins, not 6/6 (still p<0.05)
- LSAC/DP is degenerate (majority-class collapse), not a clean method loss
- LSAC accuracy under DP is pinned *at* the constant-predictor baseline, not below it —
  the "α≥0.3 below constant predictor" claim applies to Adult and Credit only
- UTKFace claims only from rows tagged `data_provenance: REAL`
- Never re-run the 540-row grid just to rebuild the paper — the JSON is committed

## Do not

- Create a working-tree `_archive/` directory — see `ARCHIVE_POLICY.md`
- Delete or move any `experiments/*.py` without grepping for references first
- Touch `results/canonical_tau1.json` or `results/utkface_canonical.json` while a
  `run_*` process is actively writing to it (`ps aux | grep run_`)
