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

### Phase 0 audit — Finding 1: which π_clean formula actually ran (2026-08-05)

**Decision (b): keep `a_val` path, document honestly.** The canonical 540-row grid
labels every row `radii_mode: "uniform"`. In code (`dro_fair.py::_compute_radii`,
lines 102-120), the branch order is:

1. `radii_mode == 'empirical'` → `_empirical_pi_clean(pi_obs)` (known-attack inversion)
2. `a_val is not None` → `pi_clean = group_proportions(a_val)` ← **this branch fires
   for every canonical "uniform" row**, because `run_single_experiment` always passes
   `a_val` (clean validation labels)
3. `else` → `pi_clean = (pi_obs − α)/(1 − 2α)` (the documented Appendix F closed form) ←
   **this branch has NEVER executed in any canonical row**

The closed-form `(π̂_obs − α)/(1 − 2α)` is **dead code in the canonical path**. Every
canonical "uniform" row actually used clean validation-set group proportions
(`np.mean(a_val == j)`), not the bias-corrected estimate from corrupted training data.

**Why decision (b) is defensible:** Both Naive and DRO have access to clean validation
data (the split is train/val/test with no leakage). Using `a_val` to estimate clean
group proportions is a legitimate design choice — it is NOT an oracle leak (the trainer
never sees clean training labels, only clean validation group *rates*). The radii
computed from `a_val` are tighter (closer to the true clean proportions) than the
closed-form, which is conservative.

**What must change in the paper:** The paper/Appendix F must NOT claim that the
canonical grid tests the theoretical closed-form `(π̂_obs − α)/(1 − 2α)`. It should
state that the canonical grid uses clean validation group proportions for radius
calibration (a practical choice, since val data is available), and that the closed-form
is a fallback for the no-clean-validation setting. The `radii_mode: "uniform"` label in
the JSON is retained for backward compatibility but should be understood as
"validation-estimated" in the paper text.

**What does NOT change:** The locked 540 rows, the headline claims, the Wilcoxon
results. The radii used were valid (from clean val data); only the *label* and the
*paper claim* about which formula ran were wrong.

### Phase 0 audit — Item 4: TV → L1 ×2 factor (2026-08-05)

**VERIFIED CORRECT.** `dro_fair.py:219`:
`project_simplex_l1_ball(p_np, center_np, 2 * radius, ...)`.
The `2 * radius` converts TV radius ρ to L1 radius 2ρ (since TV = ½·L1). Unit test
confirms: projecting with TV radius ρ=0.1 produces a point at L1 distance exactly
0.2 from center, on the ball boundary. No factor-of-2 bug.

### Phase 0 audit — Items 5-6: tilted risk and dual ascent (2026-08-05)

**Item 5 (tilted risk) VERIFIED.** `dro_fair.py:255-263`:
`β * (logsumexp(ℓ/β) − log(m))` = `β * log(mean(exp(ℓ/β)))`. Matches Algorithm 1 step 2.

**Item 6 (dual ascent) VERIFIED.** `dro_fair.py:309`: `lambda_lr = η · 0.95^epoch`
(epoch = 0-indexed training epoch). `dro_fair.py:334`:
`λ ← clamp(λ + lambda_lr · g, 0, λ_max)`. Matches Algorithm 1 step 3. The decay
counter is the epoch index (one dual update per epoch), which is correct.

### Phase 0 audit — Invariant checks (2026-08-05)

Run across all 560 rows of `canonical_tau1.json`:

| Invariant | Result |
|---|---|
| `dp_clean ≥ 0` every row | **PASS** (0 violations) |
| `if_clean ≥ 0` every row | **PASS** (0 violations) |
| `acc_clean ∈ [0,1]` every row | **PASS** (0 violations) |
| α=0 gap vs α=0.1 gap | Adult/Credit: small (0.001-0.007), consistent with tilted-risk-vs-BCE difference. **LSAC: large (+0.038)** — the known degeneracy (DRO collapses to majority predictor). Not a new bug. |

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
