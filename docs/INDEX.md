# Docs index

**Cleaned 2026-08-04** — 14 superseded/duplicate session docs removed (recoverable via
`git log --oneline --all -- <path>`; see `docs/reference/ARCHIVE_POLICY.md`). Two agents
(GLM, Grok) are working this repo alongside the assistant — their task assignments are
the two handoff docs below, kept deliberately separate so their file writes never overlap.

## Task assignments — the only two docs an agent needs to start work

| Doc | Owner |
|-----|-------|
| [HANDOFF_GLM.md](HANDOFF_GLM.md) | GLM — Mac CPU ablation queue |
| [HANDOFF_GROK.md](HANDOFF_GROK.md) | Grok — flair2 GPU work |

## Status and findings (read-only reference, current)

| Doc | Role |
|-----|------|
| [../STATUS.md](../STATUS.md) | Single source of truth |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | Claim audit — every number traced to data |
| [KULDEEP_CORRECTION.md](KULDEEP_CORRECTION.md) | Honest corrections sent to the advisor |
| [LSAC_DEGENERACY.md](LSAC_DEGENERACY.md) | LSAC/DP collapse — diagnosis + tested fix (no fix found) |
| [UTKFACE_STATUS.md](UTKFACE_STATUS.md) | UTKFace REAL 90/90 status |
| [KEY_FORMULAS.md](KEY_FORMULAS.md) | Math + canonical τ=1 protocol |

## Data truth

- `results/canonical_tau1.json` — 540-row locked tabular grid (seeds 0-5; a resume-safe
  n=10 extension may show extra rows for seeds 6-9, claims still use the original 540)
- `results/utkface_canonical.json` — 90/90 REAL UTKFace rows
- `paper/main.pdf`, `report/report.pdf` — current build

## Durable design reference

See [`reference/`](reference/) — math derivations, attack design, ablation decisions,
repo layout, ops runbooks. Not status docs; don't expect them to reflect tonight's numbers.
