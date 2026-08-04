#!/usr/bin/env python3
"""Agent L2 — LSAC degeneracy fix: hypothesis test (not tuning-until-it-wins).

HYPOTHESIS: LSAC/DP is degenerate because the DRO radii formula
    rho_dp[j] = alpha / ((1-alpha)*pi_clean[j] + alpha)
blows up on the ~90/10 imbalanced minority group. With pi_clean[minority]=0.1:
    alpha=0.1 -> rho_min=0.53  (4.8x majority's 0.11)
    alpha=0.2 -> rho_min=0.71  (3.3x majority's 0.22)
    alpha=0.3 -> rho_min=0.81  (2.5x majority's 0.32)
    alpha=0.4 -> rho_min=0.87  (2.0x majority's 0.43)
The minority group is over-weighted in the worst-case reweighting, driving the
classifier to ignore features and predict the majority class -> accuracy pins
at the 0.9016 constant-predictor baseline and DP freezes at ~0.222 for
alpha in {0.2,0.3,0.4} (canonical LSAC/DP, 6/6 seeds).

FIX UNDER TEST (clamp=0.3, chosen on PRINCIPLE before running):
    0.3 caps the minority radius at the majority-group radius level (majority
    radius is 0.11..0.43 across alpha; 0.3 sits near the majority radius at
    alpha=0.3 = 0.32). This prevents minority over-weighting while still
    allowing the minority some worst-case slack above the majority. It is NOT
    tuned — it is the smallest cap that brings the minority radius into the
    same order of magnitude as the majority radius, derived from the formula
    on the diagnosed imbalance, BEFORE seeing any L2 result.

ARMS (LSAC only, attack='dp', 5 alpha x 6 seeds):
  (a) radii_mode='uniform', radii_clamp=None  = CANONICAL LSAC reference
      -> PULLED FROM canonical_tau1.json (do NOT re-run here).
  (b) radii_mode='uniform', radii_clamp=0.3   -> clamp rho_dp to max 0.3
  (c) radii_mode='empirical', coordinated=True, radii_clamp=None  -> empirical radii
  (d) radii_mode='empirical', coordinated=True, radii_clamp=0.3   -> empirical + clamp

New DRO runs: arms (b,c,d) x 5 alpha x 6 seeds = 90 configs.
Naive baseline: naive does not use radii, so naive b=c=d -> 30 naive configs
run ONCE under the coordinated attack (the harder case matching arms c,d). Arm
(b)'s naive comparator is the canonical LSAC naive (coordinated=False), pulled
read-only from canonical_tau1.json by the summarize script.
TOTAL: 90 DRO + 30 naive = 120 configs -> results/lsac_radii_fix.json.

RULE: this is hypothesis testing, not tuning-until-it-wins. Run once, report.
  - If any arm UN-DEGENERATES LSAC (accuracy moves off 0.9016 AND DP unfreezes
    across alpha), the paper UPGRADES LSAC from "degenerate, excluded" to
    "recovered by attack-aware radius calibration".
  - If none do, the limitation stands WITH EVIDENCE instead of a hypothesis.
Both outcomes ship. Only the untested state does not.

Launch: ABLATION_WORKERS=4 (6 other drivers in flight; do NOT exceed 4).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASET = "lsac"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
ATTACK = "dp"
RADII_CLAMP = 0.3  # principled choice; see module docstring.

# DRO arms (b, c, d). Each tuple is the 16-element schema consumed by
# run_ablation_parallel._worker: (ds, alpha, seed, method, attack, k_inner,
# pgd_steps, tau, lambda_init, lr_lambda, radii_mode, coordinated,
# corruptor_type, attack_k, radii_scale, radii_clamp).
DRO_ARMS = [
    # (b) uniform, clamp=0.3  (coordinated=False = canonical corruption)
    ("uniform", False, None, 0.3),
    # (c) empirical, coordinated=True, clamp=None
    ("empirical", True, None, None),
    # (d) empirical, coordinated=True, clamp=0.3
    ("empirical", True, 0.3, 0.3),
]


def build_configs():
    configs = []
    # DRO arms b, c, d: 3 arms x 5 alpha x 6 seeds = 90
    for radii_mode, coord, _, clamp in DRO_ARMS:
        for a in ALPHAS:
            for s in SEEDS:
                configs.append((DATASET, a, s, "dro", ATTACK, 10, 20, 1.0,
                                0.0, 5e-3, radii_mode, coord, 'adversarial', 5,
                                1.0, clamp))
    # Naive baseline: 30 configs, run ONCE under coordinated=True (harder case;
    # matches arms c,d on the same corrupted data). Arm (b)'s naive comparator
    # is the canonical LSAC naive (coordinated=False), pulled read-only by the
    # summarize script. radii_mode/radii_clamp are irrelevant for naive but are
    # recorded as the arm-(c) defaults so naive keys map deterministically.
    for a in ALPHAS:
        for s in SEEDS:
            configs.append((DATASET, a, s, "naive", ATTACK, 10, 20, 1.0,
                            0.0, 5e-3, 'empirical', True, 'adversarial', 5,
                            1.0, None))
    return configs


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run("results/lsac_radii_fix.json", build_configs(),
        provenance_extras={
            "ablation": "l2_lsac_radii",
            "n_seeds_planned": 6,
            "radii_clamp_chosen": RADII_CLAMP,
            "radii_clamp_justification": (
                "LSAC minority radius blows up to 0.53..0.87 (2.0..4.8x the "
                "majority radius 0.11..0.43) because pi_clean[minority]=0.1 "
                "shrinks the denominator. 0.3 caps the minority radius at the "
                "majority-group radius level (near majority radius at alpha=0.3 "
                "=0.32), preventing minority over-weighting. Chosen on principle "
                "before running; not tuned."
            ),
        },
        workers=w, label="L2-LSAC")