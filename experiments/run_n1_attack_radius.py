#!/usr/bin/env python3
"""Agent N1 — attack strength × radius sensitivity.

Kuldeep, May 29, verbatim (his FIRST technical question, unanswered for 14
project-months): "At lower corruption levels (α=0.1): DRO does not
significantly outperform Naive — the attack is too weak to differentiate.
Does the attack affect the radius? ... if the attack is too weak, then DRO
would perform well? specially at α=0.1."

Two arms, BOTH stamping MEASURED attack effectiveness (the ΔDP the
corruption itself induces on the training labels, pre-training) as the
provenance field 'attack_effectiveness' on every row.

ARM A — vary attack strength at fixed α (vary pgd_steps):
    pgd_steps ∈ {5, 50} (canonical=20, NOT re-run here), attack='dp',
    3 datasets × α ∈ {0.1, 0.2} × 6 seeds × 2 methods × 2 pgd_steps
    = 144 configs → results/attack_strength.json
    The MEASURED attack_effectiveness field is the point: strength must be
    measured, not assumed. pgd_steps=20 rows are pulled from canonical
    (read-only) for the comparison.

ARM B — vary radius at fixed attack (vary radii_scale):
    radii_scale ∈ {0.5, 2.0} (1.0 = canonical, NOT re-run here),
    3 datasets × dp × 5 alphas × 6 seeds × DRO only × 2 radii_scale
    = 180 configs → results/radius_sensitivity.json

Both arms use ABLATION_WORKERS=4 (6 other drivers running; never >4).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["adult", "credit", "lsac"]
SEEDS = range(6)
ATTACK = "dp"

# ARM A
ALPHAS_A = [0.1, 0.2]
METHODS_A = ["naive", "dro"]
PGD_STEPS_A = [5, 50]  # canonical=20 NOT re-run

# ARM B
ALPHAS_B = [0.0, 0.1, 0.2, 0.3, 0.4]
METHOD_B = "dro"
RADII_SCALES_B = [0.5, 2.0]  # canonical=1.0 NOT re-run


def build_configs_arm_a():
    """14-tuple: (ds, a, s, m, attack, k_inner, pgd_steps, tau,
                   lambda_init, lr_lambda, radii_mode, coordinated,
                   corruptor_type, attack_k).
    pgd_steps varies in {5, 50}. canonical defaults elsewhere.
    """
    configs = []
    for ds in DATASETS:
        for a in ALPHAS_A:
            for s in SEEDS:
                for m in METHODS_A:
                    for pgd in PGD_STEPS_A:
                        configs.append(
                            (ds, a, s, m, ATTACK, 10, pgd, 1.0,
                             0.0, 5e-3, 'uniform', False, 'adversarial', 5)
                        )
    return configs


def build_configs_arm_b():
    """16-tuple: same as Arm A plus (radii_scale, radii_clamp) appended.
    radii_scale varies in {0.5, 2.0}. DRO only. radii_clamp=None (no clamp).
    """
    configs = []
    for ds in DATASETS:
        for a in ALPHAS_B:
            for s in SEEDS:
                for rs in RADII_SCALES_B:
                    configs.append(
                        (ds, a, s, METHOD_B, ATTACK, 10, 20, 1.0,
                         0.0, 5e-3, 'uniform', False, 'adversarial', 5,
                         rs, None)
                    )
    return configs


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "both"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if arm in ("a", "both"):
        print(f"\n==== ARM A: attack_strength (pgd_steps in {{5,50}}) ====", flush=True)
        run("results/attack_strength.json", build_configs_arm_a(),
            provenance_extras={"ablation": "n1_attack_strength",
                               "arm": "A",
                               "n_seeds_planned": 6},
            workers=w, label="N1-A-AttackStrength")

    if arm in ("b", "both"):
        print(f"\n==== ARM B: radius_sensitivity (radii_scale in {{0.5,2.0}}) ====", flush=True)
        run("results/radius_sensitivity.json", build_configs_arm_b(),
            provenance_extras={"ablation": "n1_radius_sensitivity",
                               "arm": "B",
                               "n_seeds_planned": 6},
            workers=w, label="N1-B-RadiusSensitivity")

    print("\nN1: both arms done.", flush=True)