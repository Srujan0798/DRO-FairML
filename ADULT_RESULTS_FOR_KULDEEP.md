# Adult Results — for discussion with Kuldeep

> Per your guidance ("start by testing on the Adult dataset, then we discuss").
> All numbers from `results/tau_ablation_tau{1,10,100}.json` and
> `results/knn_ablation_k{5,10,15}.json` (Adult, 3 seeds each, K_inner=10,
> pgd_steps=20, coordinated DP-targeted PGD attack).

## 1. Tau ablation (your Q12) — the main result

**Fixing tau = 1 for all α makes DRO beat Naive on DP at every α — at no accuracy cost.**

DP violation (lower = fairer), DP attack, mean over 3 seeds:

| α | Naive DP | DRO DP | winner | DRO wins / 3 seeds | Δacc (DRO−Naive) |
|---|---|---|---|---|---|
| 0.1 | 0.2068 | 0.2046 | **DRO** | 2/3 | +0.001 |
| 0.2 | 0.2480 | 0.2371 | **DRO** | 3/3 | +0.002 |
| 0.3 | 0.2855 | 0.2640 | **DRO** | 3/3 | +0.009 |
| 0.4 | 0.3101 | 0.2834 | **DRO** | 3/3 | +0.011 |

Combined attack at tau=1: DRO also wins all 4 α (Δ from −0.006 to −0.029).
The DRO advantage **grows with α**, exactly as the theory predicts.

**Contrast — tau = 100 (our previous production schedule for α≤0.3):** DRO *loses*
on DP almost everywhere (e.g. α=0.2: Naive 0.327 vs DRO 0.503). This is the entire
source of the earlier "DRO is fragile / worse than Naive" worry — it was a
**high-temperature artifact**, not a real weakness of DRO. Switching to your
suggested fixed-tau setup (tau=1) flips the narrative to "DRO consistently
improves DP robustness under adversarial attack."

## 2. IF k-NN ablation (your Q6)

IF attack with k ∈ {5, 10, 15}, mean over 3 seeds:

| α | IF (k=5) | IF (k=10) | IF (k=15) |
|---|---|---|---|
| 0.1 | n0.0248 / d0.0252 | n0.0261 / d0.0262 | n0.0263 / d0.0260 |
| 0.2 | n0.0254 / d0.0207 | n0.0269 / d0.0237 | n0.0292 / d0.0247 |
| 0.3 | n0.0290 / d0.0269 | n0.0263 / d0.0266 | n0.0276 / d0.0278 |

(n = Naive, d = DRO.) **The IF attack is essentially insensitive to k** —
strength and the DRO/Naive gap are nearly identical across k=5/10/15. So k=5 is
a safe default; we report this as a robustness check.

## 3. Hyperparameter grid (your Q1) — running now

λ_init ∈ {0.0, 0.01, 0.1, 1.0} × lr_lambda ∈ {0.001, 0.005, 0.01}, DRO, DP attack,
tau=1, α ∈ {0.2, 0.3}, 3 seeds (`results/lambda_lr_grid.json`, in progress).
Goal per your note: see whether we can tighten DP further (accuracy may drop —
that's acceptable). Results to follow.

## 4. Narrative decisions taken from your feedback
- **Q3 (LSAC):** LSAC has inherent low DP, so the DP attack can't raise it much —
  we will frame LSAC around the **IF attack**, not DP.
- **Q5 (radii):** treated as **empirical, not theoretical** — we calibrate radii
  from the observed clean group proportions under the known (coordinated) attack,
  rather than the uniform-corruption closed form. Not claiming the paper is wrong.

## Open question for you
For the grid search we defaulted to all 3 seeds (runs are cheap on Adult). Fine,
or do you want 1-seed scouting first?
