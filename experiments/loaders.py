"""Shared, fail-loud result loaders.

Agent C (MASTER_DISPATCH.md) requires that no loader silently fall back to a
stale / contaminated results file. The single source of truth for tabular
results is ``results/canonical_tau1.json`` (540 rows, full provenance:
tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0).

The legacy ``results/fairness_pgd_results.json`` is CONTAMINATED: 270
pre-provenance rows with ``tau=None`` / mixed config. Every attempt to read
it now raises loudly instead of silently producing wrong artifacts.
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

CANONICAL_PATH = os.path.join(RESULTS_DIR, "canonical_tau1.json")
CONTAMINATED_PATH = os.path.join(RESULTS_DIR, "fairness_pgd_results.json")


def load_canonical_tau1():
    """Load the canonical 540-row tau=1 grid. Fails loudly if absent."""
    if not os.path.exists(CANONICAL_PATH):
        raise FileNotFoundError(
            f"Canonical results missing: {CANONICAL_PATH}. "
            f"Run the canonical grid (tau=1, k_inner=10) before loading results."
        )
    with open(CANONICAL_PATH) as f:
        return json.load(f)


def load_fairness_pgd_results():
    """Legacy loader — intentionally raises.

    results/fairness_pgd_results.json is contaminated (pre-provenance rows,
    tau=None). Do NOT read it. Use load_canonical_tau1() instead.
    """
    raise RuntimeError(
        "results/fairness_pgd_results.json is CONTAMINATED (270 pre-provenance "
        "rows, tau=None) and must not be used. Load results/canonical_tau1.json "
        "(540 rows, full provenance) via load_canonical_tau1() instead."
    )
