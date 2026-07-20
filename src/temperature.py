"""Canonical training temperature τ for DRO-FAIR.

This is the single source of truth for `get_temperature`. Every experiment runner
imports it from here rather than defining its own copy.

History: the project originally used a *stepped* schedule
(`return 1.0 if alpha >= 0.4 else 100.0`) that made DRO look fragile at low
corruption levels. The tau=100 -> tau=1 fix is the correct central finding
(see docs/MASTER_DISPATCH.md). τ is therefore a constant 1.0 for every α.
"""

TAU = 1.0


def get_temperature(alpha=None):
    """Return the canonical training temperature τ.

    The argument is retained for API compatibility with callers that pass the
    corruption level α, but τ is constant (1.0) across all configurations.
    """
    return TAU
