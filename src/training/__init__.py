from .naive_fair import NaiveFairTrainer
from .dro_fair import DroFairTrainer
from .standard_ml import StandardMLTrainer

__all__ = ['NaiveFairTrainer', 'DroFairTrainer', 'StandardMLTrainer', 'get_run_config']


def get_run_config(
    k_inner: int = 10,
    tau: float = 1.0,
    radii_mode: str = "uniform",
    lambda_init: float = 0.0,
    coordinated: bool = False,
    pgd_steps: int = 20,
    epochs: int = 60,
    lambda_max: float = 1.5,
    **kwargs,
) -> dict:
    """Return a provenance dict capturing the full run configuration.

    Per MASTER_PLAN §1/§3/§5: every result row MUST record its full config
    so that mixed k_inner/tau/radii_mode etc never silently contaminate data.

    Trivial for Agent A (runners owner) to import and use:
        from src.training import get_run_config
        row = {...}
        row.update(get_run_config(k_inner=10, tau=1.0, radii_mode=radii_mode,
                                  lambda_init=0.0, coordinated=False,
                                  pgd_steps=20, epochs=60, ...))
        # or row['config'] = get_run_config(...)

    This is a pure helper (no side effects); all fields align with paper spec
    (K_inner=10, lambda_init=0.0 default except Q1 ablation, lambda_max=1.5,
    tau=1 for canonical per verified headline, radii_mode for Q5, etc).
    Extra kwargs (e.g. n_seeds_planned, use_dp, use_if) are merged in.
    """
    config = {
        "k_inner": k_inner,
        "tau": tau,
        "radii_mode": radii_mode,
        "lambda_init": lambda_init,
        "coordinated": coordinated,
        "pgd_steps": pgd_steps,
        "epochs": epochs,
        "lambda_max": lambda_max,
    }
    config.update(kwargs)
    return config
