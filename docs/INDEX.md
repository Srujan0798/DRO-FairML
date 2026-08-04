# Docs index

**Cleaned 2026-08-04** — 14 superseded/duplicate session docs removed (recoverable via
`git log --oneline --all -- <path>`; see `docs/reference/ARCHIVE_POLICY.md`). Two agents
(GLM, Grok) are working this repo alongside the assistant — their task assignments are
the two handoff docs below, kept deliberately separate so their file writes never overlap.

## Task assignments — the only two docs an agent needs to start work

**Both docs open with a PHASE 0 correctness audit — first-principles verification of
the actual theory/formulas, not new experiments. Direct instruction: don't rush this,
resume Phase 1 (new experiments) only once Phase 0's checkboxes are done.**

| Doc | Owner | Phase 0 covers |
|-----|-------|-----------------|
| [HANDOFF_GLM.md](HANDOFF_GLM.md) | GLM | Training/radii math — the "uniform" radii formula appears to have never executed in any canonical row (Finding 1), the TV→L1 ×2 factor is unverified, tilted-risk/dual-ascent formulas need hand re-derivation |
| [HANDOFF_GROK.md](HANDOFF_GROK.md) | Grok | Metrics/attack math — α=0 disclosure, IF cosine-metric disclosure, IF attack/eval k-NN graph match, IF→DP coupling argument |

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
