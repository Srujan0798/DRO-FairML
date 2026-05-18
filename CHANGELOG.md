# Changelog

All notable fixes and improvements to the DRO-FAIR project.

## [v1.0.1] — 2026-05-18 — Audit Fixes

### Fixed
- **Credit IF claim inflated:** Report and README claimed Credit α=0.3 IF reduction was −100%, but `results/all_results.json` computes to −96.0%. Fixed to −96% in `report/report.tex`, `README.md`, and regenerated PDFs.
- **README IF win count ambiguous:** README claimed "IF in 7/9 cells" without specifying mean-based. Changed to "IF in **5/9 cells** (Wilcoxon p<0.05; mean-based: DP 6/9, IF 7/9)".
- **PDFs stale:** `report/report.pdf` and `submission/report.pdf` were NOT recompiled after the tex fix. Recompiled with `tectonic`.
- **Runtime numbers inconsistent:** Recent commits standardized everything to ~54× overhead, but authoritative `results/all_results.json` computes to **37.5×**. Fixed report table, README, and `results/runtime_table.tex` to match data.
- **Dead code stale:** `experiments/generate_pdf_report.py` still cited 7/9 IF. Updated to 5/9.
- **validate_results.py used mean-only comparison:** Rewrote to use **Wilcoxon signed-rank test** to match report methodology.
- **results/runtimes.json stale:** Had 95 experiments and 66× overhead. Regenerated from full 150-experiment dataset (37.5×).
- **setup.py outdated:** URL pointed to anonymous repo; Python classifiers missing 3.12–3.14. Fixed.
- **Makefile broken targets:** `monitor` and `review` referenced non-existent scripts. Updated with working alternatives.

### Added
- `CHANGELOG.md` — This file.
- `docs/PROJECT_REFERENCE.md` — Project facts, file structure, hyperparameters, conventions.
- `docs/KEY_FORMULAS.md` — Math reference and core formulas.
- `docs/IMPLEMENTATION_NOTES.md` — Design decisions, derivations, and code map.
- `docs/REVIEW_CHECKLIST.md` — Self-review checklist with grading rubric.
- `docs/VERIFICATION_PROTOCOL.md` — Step-by-step reproducibility verification.
- `docs/FAQ.md` — Frequently asked questions about the implementation.
- `submission/MANIFEST.md` — Submission package inventory.
- `tests/conftest.py` — Pytest configuration to prevent plugin hangs.
- `.github/workflows/verification.yml` — CI verification step.
- `PRE_SUBMISSION_CHECKLIST.md` — Pre-submission verification checklist.

## [v1.0.0] — 2026-05-16 — Initial Release

### Features
- Full implementation of DRO-FAIR Algorithm 1 from ICML submission
- Multi-modal adversarial corruption (PGD features, coordinated label flips, minority-targeted attribute flips)
- 150 experiments across Adult, Credit, LSAC
- 32 unit tests + 8 theoretical verification checks
- 7 publication-quality figures
- LaTeX report with ICML-style typography

### Stability Fixes (vs Paper)
- `lambda_max`: 2.0 → 1.5 (prevents λ runaway)
- `tau_warmup_epochs`: 10 → 15 (prevents early collapse)
- `grad_clip`: 1.0 → 0.5 (prevents gradient explosion)
- Added `lambda_decay`: 0.95^epoch (gradual regularization)

### Known Limitations
- Adult α≥0.3: DRO-FAIR collapses due to adversarial feedback loop (documented in report Section 5)
- Runtime overhead ~37.5× on CPU (paper reports ~12× on GPU)
- Full-batch training limits scalability
