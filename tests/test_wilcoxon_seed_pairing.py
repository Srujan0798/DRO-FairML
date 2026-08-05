"""Regression: Wilcoxon tables must pair Naive/DRO by seed, not row order."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def test_adult_if_alpha02_dp_pvalue_matches_seed_paired():
    """Historical bug: unpaired lists reported p≈0.344 instead of 0.0156.

    Locked to the original n=6 slice (seeds 0-5), same as
    test_adult_dp_alpha01_is_five_of_six below. Agent S appends seeds 6-9 to
    results/canonical_tau1.json for the N10 extension, so reading the raw file
    unfiltered would silently change n and the expected p-value. The paper
    itself is unaffected: experiments/loaders.load_canonical_tau1() returns
    seeds 0-5 unless include_extension=True, which is why the generator
    assertion further down still holds at n=6.
    """
    rows = json.loads((ROOT / "results" / "canonical_tau1.json").read_text())
    by_seed = {}
    for r in rows:
        if (
            r["dataset"] == "adult"
            and r["attack"] == "if"
            and abs(float(r["alpha"]) - 0.2) < 1e-9
            and int(r["seed"]) < 6
        ):
            by_seed.setdefault(int(r["seed"]), {})[r["method"]] = r

    seeds = sorted(by_seed)
    assert seeds == [0, 1, 2, 3, 4, 5]
    nv = np.array([by_seed[s]["naive"]["dp_clean"] for s in seeds])
    dv = np.array([by_seed[s]["dro"]["dp_clean"] for s in seeds])
    _, p = wilcoxon(nv - dv, alternative="greater")
    assert abs(p - 0.015625) < 1e-6

    # Generator must agree
    from experiments.generate_report_tables import build_wilcoxon, load_canonical_rows

    wdf = build_wilcoxon(load_canonical_rows())
    cell = wdf[
        (wdf["dataset"] == "adult")
        & (wdf["attack"] == "if")
        & (abs(wdf["alpha"] - 0.2) < 1e-9)
    ].iloc[0]
    assert abs(float(cell["dp_pvalue"]) - 0.015625) < 1e-6


def test_canonical_loader_excludes_n10_extension_by_default():
    """The paper's tables must stay at the locked n=6, whatever is on disk.

    results/canonical_tau1.json is append-only and currently holds a partial
    n=10 extension (seeds 6-9 present for some adult cells only). If the loader
    ever started returning those rows by default, every generated table would
    silently switch to a mixed-n protocol -- some cells n=10, others n=6 --
    changing published p-values with no visible signal. This asserts the
    default stays seeds 0-5, and that opting in is what actually widens it.
    """
    from experiments.loaders import load_canonical_tau1

    locked = load_canonical_tau1()
    assert {int(r["seed"]) for r in locked} <= {0, 1, 2, 3, 4, 5}

    extended = load_canonical_tau1(include_extension=True)
    assert len(extended) >= len(locked)
    # Every generated Wilcoxon cell must be exactly n=6 under the default load.
    from experiments.generate_report_tables import build_wilcoxon

    assert set(build_wilcoxon(locked)["n_pairs"]) == {6}


def test_ablation_driver_refuses_locked_science_paths():
    from experiments.run_ablation_parallel import _assert_safe_results_path
    import pytest

    with pytest.raises(RuntimeError):
        _assert_safe_results_path("results/canonical_tau1.json")
    with pytest.raises(RuntimeError):
        _assert_safe_results_path("results/utkface_canonical.json")
    # Separate ablation path is allowed
    _assert_safe_results_path("results/knn_ablation.json")


def test_adult_dp_alpha01_is_five_of_six():
    """Locked to the original n=6 baseline (seeds 0-5).

    Agent S is appending seeds 6-9 to results/canonical_tau1.json for the n=10
    extension (append-only; the original 540 rows are byte-identical, verified
    separately). This test asserts an invariant of that original n=6 slice, so
    it must not pick up the new seeds — do that in a separate n=10 test once
    Agent S's extension is complete and verified.
    """
    rows = json.loads((ROOT / "results" / "canonical_tau1.json").read_text())
    by_seed = {}
    for r in rows:
        if (
            r["dataset"] == "adult"
            and r["attack"] == "dp"
            and abs(float(r["alpha"]) - 0.1) < 1e-9
            and int(r["seed"]) < 6
        ):
            by_seed.setdefault(int(r["seed"]), {})[r["method"]] = r
    assert sorted(by_seed) == [0, 1, 2, 3, 4, 5]
    wins = sum(
        by_seed[s]["dro"]["dp_clean"] < by_seed[s]["naive"]["dp_clean"]
        for s in by_seed
    )
    assert wins == 5
    nv = np.array([by_seed[s]["naive"]["dp_clean"] for s in sorted(by_seed)])
    dv = np.array([by_seed[s]["dro"]["dp_clean"] for s in sorted(by_seed)])
    _, p = wilcoxon(nv - dv, alternative="greater")
    assert abs(p - 0.03125) < 1e-6
