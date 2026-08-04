# Docs index (canonical set)

## Start here
| Doc | Purpose |
|-----|---------|
| [../README.md](../README.md) | Project overview + reproduce |
| [../STATUS.md](../STATUS.md) | Single source of truth (status) |
| [MEETING_2026-08-04.md](MEETING_2026-08-04.md) | 4pm meeting brief + real IF |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | Claim → data audit |

## Science & design
| Doc | Purpose |
|-----|---------|
| [KEY_FORMULAS.md](KEY_FORMULAS.md) | Math |
| [FAIRNESS_PGD_DESIGN.md](FAIRNESS_PGD_DESIGN.md) | Attack design |
| [LSAC_DEGENERACY.md](LSAC_DEGENERACY.md) | Honest LSAC/DP failure |
| [KULDEEP_CORRECTION.md](KULDEEP_CORRECTION.md) | Claim corrections |
| [TAU1_ABLATION_SUMMARY.md](TAU1_ABLATION_SUMMARY.md) | Why τ=1 |
| [ABLATION_STATUS_REPORT.md](ABLATION_STATUS_REPORT.md) | Ablation adjudication |

## Pipeline & ops
| Doc | Purpose |
|-----|---------|
| [FINAL_COMPLETION_PLAN.md](FINAL_COMPLETION_PLAN.md) | Deadlines & agents |
| [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) | flair2 GPU |
| [UTKFACE_PIPELINE.md](UTKFACE_PIPELINE.md) | Image pipeline design |
| [UTKFACE_STATUS.md](UTKFACE_STATUS.md) | Live UTKFace progress |
| [COORDINATION.md](COORDINATION.md) | Multi-agent data safety |

## Repo quality
| Doc | Purpose |
|-----|---------|
| [REPO_AUDIT.md](REPO_AUDIT.md) | Full inventory |
| [REPO_CONSOLIDATION_PLAN.md](REPO_CONSOLIDATION_PLAN.md) | Cleanup stages |
| [REPO_LAYOUT.md](REPO_LAYOUT.md) | One-screen tree map |
| [PAPER_FINALIZATION_CHECKLIST.md](PAPER_FINALIZATION_CHECKLIST.md) | Paper checklist |
| [LOOP_STATUS.md](LOOP_STATUS.md) | 5-min cleanup-loop board |

## Archive
Superseded agent prompts, partial analyses, hardcoded-number hunt artifacts, and chat exports live in `_archive/` (see `_archive/cleanup_2026-08-04/`).

## Data truth
All numeric claims → `results/canonical_tau1.json` (540 rows) and `results/if_wilcoxon_summary.txt`.
