# DRO-FAIR: Key Formulas and Implementation Reference

Quick reference for the core math of DRO-FAIR (Algorithm 1) as implemented here.

**Canonical protocol (locked for Aug 10):** τ = **1.0** fixed, K_inner = 10, epochs = 60,
pgd_steps = 20, n_seeds = 6, radii_mode = uniform. Science file:
`results/canonical_tau1.json` (**540** rows). Do **not** retrain for claims.

---

## Locked headline (not the old 150-seed story)

| Claim | Status |
|-------|--------|
| Adult & Credit, α ≤ 0.2, DP + Combined | DRO lower DP vs Naive (Wilcoxon p &lt; 0.05, n=6); Adult/DP α=0.1 is **5/6** |
| IF attack | **Mixed** — see `results/if_wilcoxon_summary.txt` |
| LSAC/DP | **Degenerate** (pinned ~majority acc; DRO 0/6) |
| α ≥ 0.3 | Below constant predictor on **Adult/Credit only** (no method claim) |
| UTKFace | REAL 90/90 feature pilot; **mixed** clean-test (`results/utkface_summary.md`) |

Superseded: any “150 experiments / n_seeds=10 / stepped τ=100” headline. Those were pre-fix.

---

## Algorithm 1 step order

Each training epoch executes:

1. **Forward pass.** `h̃ = σ(τ · f_θ(x))` with **fixed τ = 1** in the canonical grid  
   (the old stepped schedule τ=100 for α≤0.3, τ=1 at α=0.4 was the artifact that made DRO look fragile).
2. **Outer minimization (θ).** AdamW on `L_tilt + λ_DP·g_DP + λ_IF·g_IF`, grad clip 0.5.
3. **Dual ascent (λ).** `λ ← clamp(λ + η_λ · 0.95^t · g, 0, λ_max=1.5)`.
4. **Inner maximization (p).** K=10 ascent steps on `∇g` (not `λ∇g` — see note), then Dykstra onto Δ_n ∩ B_1(p̂, 2ρ).

Source: `src/training/dro_fair.py`. Temperature helper: `src/temperature.py` (canonical path returns 1.0).

**Note on `∇g` vs `λ∇g`.** For λ > 0, `argmax g(p) = argmax λg(p)`. Scaling by λ inflates gradient magnitude as λ → λ_max and can destabilize the inner loop; ∇g keeps steps better conditioned.

---

## Core theoretical formulas

| Quantity | Formula | Origin |
|----------|---------|--------|
| DP radius | `ρ_DP,j = α / ((1 − α)·π_j + α)` | Theorem 4.2 |
| IF radius | `ρ_IF = 2α − α²` | Theorem 4.3 |
| Bias correction | `π_clean = (π̂_obs − α) / (1 − 2α)` | Appendix F |
| TV → L1 conversion | `L1 radius = 2 · ρ_TV` | Standard identity |

The bias correction solves `π̂_obs = (1−α)·π_clean + α·(1−π_clean)` for `π_clean`.  
L1-radius identity: `TV = ½ · L1`.

**Empirical radii (Q5):** when attack structure is known, `radii_mode='empirical'` can tighten π_clean from post-attack group proportions. Canonical 540 uses **uniform**. Derivation: `docs/reference/Q5_derivation.md` / report appendix.

---

## Design decisions worth knowing

### `λ_max = 1.5` (paper often uses 2.0)
Empirical stability: lower clamp reduces λ_DP runaway on Adult under high α.

### Temperature τ
- **Canonical claims:** fixed **τ = 1**.
- **Historical artifact:** stepped τ=100 (α≤0.3) → τ=1 (α=0.4) made DRO lose on Adult DP; that is *not* the paper story anymore.
- Softmax temperature controls how sharp predictions are for fairness gradients / inner reweighting.

### Statistical reporting
Paired Wilcoxon signed-rank, one-sided H₁: Naive DP > DRO DP, **n = 6** seeds on the locked grid. Report win counts (including honest **5/6** cells).

### Theorem 6.1 under adversarial corruption
PGD respects the αn sample budget, so TV-ball containment holds theoretically. Empirically, radii are more useful when baseline DP is not near zero (LSAC/DP is a different, degenerate failure mode).

---

## Reproduction snippet (canonical file only)

```python
# Verify any (dataset, alpha, attack) cell from the locked flat JSON
python3 -c "
import json, numpy as np
from scipy.stats import wilcoxon
d = json.load(open('results/canonical_tau1.json'))
ds, a, atk = 'adult', 0.2, 'dp'
rows = [r for r in d if r['dataset']==ds and r['attack']==atk and abs(r['alpha']-a)<1e-9]
ndp = [r['dp_clean'] for r in rows if r['method']=='naive']
ddp = [r['dp_clean'] for r in rows if r['method']=='dro']
print(f'n={len(ndp)}  Naive: {np.mean(ndp):.4f}  DRO: {np.mean(ddp):.4f}')
print(f'Wilcoxon p: {wilcoxon(ndp, ddp, alternative=\"greater\")[1]:.4f}')
"
```

Prefer project scripts: `make validate`, `make wilcoxon`, `python3 experiments/meeting_summary.py`.

---

## See also

- `STATUS.md` — SSOT
- `docs/LSAC_DEGENERACY.md`, `docs/KULDEEP_CORRECTION.md`
- `results/if_wilcoxon_summary.txt`, `results/utkface_summary.md`
