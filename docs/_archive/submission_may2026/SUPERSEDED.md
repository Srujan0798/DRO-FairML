# SUPERSEDED — submission/ fork (May 2026)

This directory is a **stale fork** of the project and must not be used.

## Why it was moved here

- It contains a copy of `src/` (14 files) of which **7 had diverged** from the live
  `src/` at the time it was archived.
- It was still on the **retracted `tau=100` defaults**, not the canonical `tau=1.0`.
- It ships `report.pdf` asserting the **retracted "DRO is fragile" conclusion** that the
  project later disproved.

Anyone who opened this directory first would have received the wrong science. It is kept
here only for historical reference, under archival (not source) status.

## Correct path

Use the live `src/`, `experiments/`, `paper/`, and `report/` at the repository root.
The current central finding is that the `tau=100 → tau=1` fix is correct and that DRO
wins on Adult and Credit at `alpha <= 0.2` under the canonical config.

## What was in here

- `run_experiments.py` — a duplicate runner, also on the stepped-tau defaults.
- `src/` — the diverged copy.
- `report.pdf` — the retracted finding.
- `*.png`, `*.csv`, `all_results.json` — stale results/figures.
