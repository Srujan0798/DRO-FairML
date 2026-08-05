"""Generated tables must never print a significance star on a degenerate metric.

Under the DP and COMBINED attacks the canonical grid records an IF violation of
~1e-11 -- floating-point noise from the pre-fix Euclidean k-NN training graph,
not a measured quantity. Running Wilcoxon on that noise previously produced two
cells printed as significant (p=0.016***) beside a 0.0% effect size, in the
built PDF. These tests pin the fix: such cells render as "--", and no table row
may ever pair a significance star with a degenerate IF mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.generate_report_tables import (  # noqa: E402
    IF_DEGENERATE_TOL,
    _format_if_cell,
    build_wilcoxon,
    load_canonical_rows,
)


def test_degenerate_if_cell_renders_as_dashes():
    row = {'if_naive_mean': 7.2e-11, 'if_diff_mean': 4.9e-11, 'if_pvalue': 0.015625}
    assert _format_if_cell(row) == "-- & --"


def test_non_degenerate_if_cell_still_reports_normally():
    row = {'if_naive_mean': 0.0457, 'if_diff_mean': 0.0091, 'if_pvalue': 0.0156}
    out = _format_if_cell(row)
    assert "--" not in out
    assert "***" in out          # genuinely significant, genuinely measured
    assert "19.9" in out         # 0.0091/0.0457 ~ 19.9%


def test_no_canonical_row_pairs_a_star_with_a_degenerate_if_mean():
    """The actual defect that reached the PDF: *** printed on ~1e-11 noise."""
    wdf = build_wilcoxon(load_canonical_rows())
    offenders = [
        (r['dataset'], r['alpha'], r['attack'])
        for _, r in wdf.iterrows()
        if r['if_naive_mean'] <= IF_DEGENERATE_TOL and "***" in _format_if_cell(r)
    ]
    assert offenders == [], f"significance star on degenerate IF: {offenders}"


def test_dp_and_combined_rows_are_all_degenerate_as_documented():
    """Guards the claim made in the paper's limitations section.

    If a future re-run under the corrected cosine metric fixes these rows, this
    test fails loudly -- at which point the paper's disclosure must be updated
    to say the gap is closed, rather than left claiming a limitation that no
    longer exists.
    """
    wdf = build_wilcoxon(load_canonical_rows())
    affected = wdf[wdf.attack.isin(['dp', 'combined'])]
    assert len(affected) == 30
    assert (affected['if_naive_mean'] <= IF_DEGENERATE_TOL).all(), (
        "canonical DP/COMBINED rows now carry real IF values -- the grid was "
        "re-run under the corrected metric. Update the limitations section."
    )
