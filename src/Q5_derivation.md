# Q5: Empirical π_clean Inversion Derivation (for radii_mode='empirical')

**Context (Kuldeep Q5, MASTER_PLAN §5, global constraints):**  
DRO (when `radii_mode='empirical'`) knows only `α` and the *known attack structure* (coordinated 70%-minority attribute-flip PGD). It never receives the true per-sample corruption mask (no oracle leak). This is used in `_compute_radii` / `_empirical_pi_clean` as an alternative to clean `a_val` proportions. `RandomCorruptor` is never used for the method — only `FairnessTargetedPGD(coordinated=True)`.

**Attack model (exact budget, attribute flips only):**  
- Binary protected attribute A ∈ {0,1}.  
- Let π_clean = (π_min, π_maj) be the *pre-attack* (clean) group proportions in the training distribution. N_min = π_min * n, N_maj = π_maj * n.  
- Corruption budget: exactly n_c = α * n samples are selected for attribute flip.  
- Coordinated targeting (70/30 split):  
  - 70% of budget (0.7 n_c) chosen from the *clean minority group* and flipped (min → maj).  
  - 30% of budget (0.3 n_c) chosen from the *clean majority group* and flipped (maj → min).  

**Effect on observed (post-attack) proportions:**  
Observed minority count after flips:  
N_min_obs = N_min - 0.7 n_c + 0.3 n_c = N_min - 0.4 n_c  

Observed majority count:  
N_maj_obs = N_maj - 0.3 n_c + 0.7 n_c = N_maj + 0.4 n_c  

Divide by total n:  
π_obs[min] = π_clean[min] - 0.4 α  
π_obs[maj] = π_clean[maj] + 0.4 α  

**Inversion (solve for clean proportions):**  
π_clean[min] = π_obs[min] + 0.4 α  
π_clean[maj] = π_obs[maj] - 0.4 α  

In implementation (`_empirical_pi_clean`):  
- Identify the observed minority as `minority_idx = int(np.argmin(pi_obs))` (the smaller of the *observed* groups).  
- Apply the ±0.4α correction using that index.  
- `pi_clean = np.clip(pi_clean, 0.0, 1.0)`  
- Renormalize: `pi_clean /= pi_clean.sum()` (in case of floating point or clipping at extreme α).  
- Special case α=0: return π_obs verbatim.  

**Why argmin on observed works:** For α ≤ 0.4 and realistic base rates (e.g. Adult ~0.3 minority), the net -0.4α shift does not invert which group is the smaller one. Thus the post-attack observed min is still the original min, but the formula itself does not require knowing the original labels — it just inverts using whichever side is smaller after attack.

**Comparison to uniform mode (the default):**  
Uniform (generic bias correction, no attack structure):  
π_clean[j] = (π_obs[j] - α) / (1 - 2α)   [for α ≠ 0.5, clipped]  

This assumes the α budget is distributed uniformly / independently of group. Empirical exploits the specific 70/30 coordinated bias to get a tighter (exact under model) π_clean, hence better calibrated ρ_dp = α / ((1-α)π_clean + α) for the TV balls in DRO.

**Test evidence (unit-level, no data leakage):**  
See `tests/test_radii_calibration.py:test_empirical_mode_recovers_clean_proportions` (and siblings).  
Synthesizes clean a (70/30), applies exact coordinated flips for given α (using only the structure), computes π_obs, calls `_empirical_pi_clean(pi_obs)` on a `DroFairTrainer(alpha=..., radii_mode='empirical')` instance (the inversion sees *only* π_obs + self.alpha), asserts `np.allclose(pi_clean, true_π, atol=1e-6)`.

Example (n=1000, α=0.2, clean π=(0.7,0.3)):  
- n_c=200, 140 min flips + 60 maj flips → π_obs ≈ (0.78, 0.22)  
- Inversion: min_idx=1 → π_clean[1] = 0.22 + 0.08 = 0.30; π_clean[0] = 0.78 - 0.08 = 0.70. Exact.

**Implementation constraints respected:**  
- `alpha>0` guard in fit/K_inner.  
- `epochs=60`, `K_inner=10` defaults, `lambda_max=1.5`, lambda_init=0.0.  
- Empirical radii only affect ρ (via π_clean); the p inner-max still ascends ∇g (direct on fairness violation), never λ∇g.  
- No use of true corruption mask at any point for the empirical path.

This derivation was produced by Agent B (core code & theory) and is intended for D to incorporate into appendix / paper/auto_generated/.

**References in code:**  
- `src/training/dro_fair.py` : `_empirical_pi_clean`, `_compute_radii`, module docstring, fit().  
- `tests/test_radii_calibration.py` : synthesis + recovery asserts + end-to-end with `FairnessTargetedPGD(coordinated=True)`.  
- Provenance helper `src/training/__init__.py:get_run_config` (includes `radii_mode`).  
- See also end-to-end tests and metrics for DP/IF.

(Fragment suitable for .tex inclusion; equations above are in math mode ready form.)
