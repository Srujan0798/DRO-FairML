"""Shared, fail-loud result loaders.

Agent C (MASTER_DISPATCH.md) requires that no loader silently fall back to a
stale / contaminated results file. The single source of truth for tabular
results is ``results/canonical_tau1.json``: the canonical tau=1.0, k_inner=10,
epochs=60, pgd_steps=20, lambda_init=0.0 grid. The DP and Combined attacks are
complete (360 rows, 6 seeds each); the IF-attack third (180 rows) is pending a
cluster re-run after the IF-metric fix (Agent A) and is intentionally absent.

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

# Documented majority-class (constant-predictor) accuracies, used only as a
# fallback when the raw data cannot be loaded.
_CONSTANT_PREDICTOR_FALLBACK = {
    "adult": 0.7521,
    "credit": 0.7788,
    "lsac": 0.9016,
}


def load_canonical_tau1():
    """Load the canonical tau=1 grid. Fails loudly if absent or polluted."""
    if not os.path.exists(CANONICAL_PATH):
        raise FileNotFoundError(
            f"Canonical results missing: {CANONICAL_PATH}. "
            f"Run the canonical grid (tau=1, k_inner=10) before loading results."
        )
    with open(CANONICAL_PATH) as f:
        rows = json.load(f)
    if not all(r.get("k_inner") == 10 for r in rows):
        bad = sorted({r.get("k_inner") for r in rows if r.get("k_inner") != 10})
        raise AssertionError(
            f"canonical_tau1.json contains non-k_inner=10 rows (k_inner={bad}). "
            f"It must hold only the canonical grid, not kNN-ablation rows."
        )
    return rows


def load_fairness_pgd_results():
    """Legacy loader — intentionally raises.

    results/fairness_pgd_results.json is contaminated (pre-provenance rows,
    tau=None). Do NOT read it. Use load_canonical_tau1() instead.
    """
    raise RuntimeError(
        "results/fairness_pgd_results.json is CONTAMINATED (270 pre-provenance "
        "rows, tau=None) and must not be used. Load results/canonical_tau1.json "
        "(canonical grid) via load_canonical_tau1() instead."
    )


def constant_predictor_acc(dataset):
    """Majority-class (constant-predictor) accuracy for ``dataset``.

    Computed from the data when available; falls back to the documented
    per-dataset value otherwise. Replaces the single hardcoded 0.752 that was
    previously applied to every dataset (wrong for Credit and LSAC).
    """
    d = (dataset or "").lower()
    try:
        import numpy as np
        from src.data.datasets import get_dataset

        out = get_dataset(d, data_dir=os.path.join(ROOT, "data", "raw"))
        y_train = np.asarray(out[1]).ravel().astype(int)
        counts = np.bincount(y_train)
        return float(counts.max() / counts.sum())
    except Exception:
        return _CONSTANT_PREDICTOR_FALLBACK.get(d, 0.7521)
