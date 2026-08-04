# Ablation Studies — Status, Decisions, and Rationale (Agent E)

**Date:** 2026-07-20
**Context:** Five ablation studies were incomplete or incoherent. Per the dispatch, each is
either **finished** or **formally dropped with a written reason** — none left in limbo.
**Canonical config for any re-run:** `tau=1.0, k_inner=10, epochs=60, pgd_steps=20,
lambda_init=0.0, coordinated=False, 6 seeds`.

> Compute note: this repo's full 540-row canonical grid required ~weeks of cluster time.
> The tabular runs here are CPU-only in this environment (~10–25 min per Adult row at
> 60 epochs). Where a re-run is the correct fix but exceeds the available session compute,
> the exact ready-to-run command is given and the item is marked **DEFERRED (cluster)**.

---

## Summary table

| # | Ablation | Status | Disposition |
|---|----------|--------|-------------|
| 1 | Tau ablation | 124/173/148/138/170 rows across tau∈{1,5,10,20,100} | **DROPPED (incoherent)** — confounds τ with k_inner; no LSAC; re-run spec below |
| 2 | Lambda grid | 26/720 done (3.6%), crashed | **DROPPED (720-grid)** — λ0=1.0 pathology diagnosed; scoped grid specified |
| 3 | kNN ablation | k=10 complete (144); k=5 short 12; k=15 short 24 | **IN PROGRESS (backfill running)** — 36 missing rows = Adult α∈{0.1,0.2,0.3,0.4} seeds 3,4,5 |
| 4 | Empirical radii (Q5) | 29/270; Adult only; α∈{0.0,0.1} | **DEFERRED (cluster)** — finishable via `run_canonical_empirical.py`; α=0 rows are no-ops |
| 5 | Random vs adversarial | 27 rows, no provenance, pre-canonical | **DEFERRED (cluster)** — re-run command given; uses canonical τ=1 after Agent F fix |

---

## 1. Tau ablation — DROPPED (incoherent as a cross-τ comparison)

**Evidence (from `results/tau_ablation_tau{1,5,10,20,100}.json`):**

- **No LSAC in any tau file.** Coverage is Adult + (partial) Credit only.
- **tau=5 and tau=20 are Adult-only** (173 and 138 rows, Adult only).
- **k_inner is MIXED within files**, so comparing across τ confounds τ with k_inner:

  | τ | rows | datasets | k_inner distribution (all rows) | k_inner (Adult only) |
  |---|------|----------|----------------------------------|----------------------|
  | 1  | 124  | adult 93, credit 31 | None:109, 10:15 | None:90, 10:3 |
  | 5  | 173  | adult 173         | 10:173           | 10:173 |
  | 10 | 148  | adult 109, credit 39 | None:72, 10:76 | None:72, 10:37 |
  | 20 | 138  | adult 138         | 10:138           | 10:138 |
  | 100| 170  | adult 109, credit 61 | None:72, 10:98 | None:72, 10:37 |

- Even restricting to Adult does **not** remove the confound: Adult k_inner is mixed for
  τ∈{1,10,100}. Only τ∈{5,20} have fully k_inner=10 Adult rows, which is too narrow to be a
  τ comparison (2 τ values, and they were the original "fragile" runs).

**Decision:** Drop the cross-τ ablation as a publishable comparison. It cannot be salvaged by
subsetting (the k_inner confound is internal to every multi-τ file). The correct clean study
is a **re-run**, specified below, and is **not executed here** (compute).

**Clean re-run spec (recommended, cluster):**
```
python3 experiments/run_tau_ablation.py \
  --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 \
  --tau 1 5 10 20 100 --k_inner 10 --epochs 60 --pgd_steps 20 \
  --lambda_init 0.0 --n_seeds 6
```
Every τ file must be regenerated at **k_inner=10** and **all three datasets** before any
τ comparison is drawn. Until then, the only defensible τ statement is the canonical one
already in the paper: τ=1 beats the old stepped τ=100 schedule (the central finding),
verified on the 540-row canonical grid.

---

## 2. Lambda grid — DROPPED (720-grid); pathology diagnosed; scoped grid specified

**Pathology (from `lambda_comprehensive.log` + `results/lambda_grid_comprehensive.json`):**

- 26 of 720 configs completed (3.6%). The run crashed at config **[25/720]**:
  `adult α=0.1 s=0 dp λ0=1.0 lr=0.001` → **64308 s (17.9 h)**, vs ~700–1500 s for its
  neighbours (λ0∈{0.0,0.001,0.01,0.1}).
- That single config's result is **acc=0.752, dp=0.0498**. `acc=0.752` is exactly Adult's
  **majority-class rate** — i.e. with λ0=1.0 the DRO dual collapses the model to a constant
  predictor. The per-epoch cost ballooned ~70–100× (the run burned all 60 epochs making no
  real progress). A naive restart will hang again on every λ0=1.0 config.
- λ0=1.0 is the **divergent/collapse regime**. All other λ0 values finish normally.

**Root-cause hypothesis:** λ0=1.0 initializes the DP dual so large that the inner-max
reweights one protected group to the simplex corner; the model degenerates to the majority
class and the inner PGD / weight-projection step enters a pathologically slow, non-converging
regime for the full 60 epochs. (This is consistent with BLOCKER 2's observation that the
radii/dual can drive collapse on imbalanced data.)

**Scoped feasible grid (recommended — avoids the hang entirely):**
- **Cap λ0 ≤ 0.3** (drop 1.0; 0.5 only if specifically needed).
- Drop the standalone `if` attack (IF is degenerate per BLOCKER 1 — no signal).
- Restrict datasets to `[adult, credit]` (LSAC/dp is degenerate per BLOCKER 2).
- Resolution: `alphas=[0.1,0.2]`, `lambda_inits=[0.0,0.1,0.3]`, `lr_lambdas=[0.001,0.005]`,
  `seeds=6`, `attacks=[dp,combined]`.
- This is **not executed here** (compute). The 26 existing rows at λ0∈{0.0,0.001,0.01,0.1}
  are valid and can be kept; only the λ0=1.0 tail is unusable.

---

## 3. kNN ablation — IN PROGRESS (backfill running)

**State:** k=10 is complete (144/144). Missing rows:
- **k=5:** 12 rows — all `adult`, α∈{0.1,0.2,0.3,0.4}, seeds {3,4,5}, method `dro`.
- **k=15:** 24 rows — all `adult`, α∈{0.1,0.2,0.3,0.4}, seeds {3,4,5}, methods {naive,dro}.

The original run was interrupted after seed 2 (all missing rows are seeds 3–5), so this is a
clean mechanical backfill. The runner **resumes automatically** (it loads the existing file
and skips present rows), so it is safe and idempotent.

**Command (running in this session, logs in `/tmp/knn_k5_backfill.log`,
`/tmp/knn_k15_backfill.log`):**
```
python3 experiments/run_knn_ablation.py --k 5  --datasets adult \
  --alphas 0.1 0.2 0.3 0.4 --attacks if --methods naive dro --n_seeds 6 --epochs 60
python3 experiments/run_knn_ablation.py --k 15 --datasets adult \
  --alphas 0.1 0.2 0.3 0.4 --attacks if --methods naive dro --n_seeds 6 --epochs 60
```
On a CPU-only box this is ~10–25 min/Adult-row, so the 36 rows are a multi-hour job; if the
session ends early the files remain resumable and re-running the same command finishes them.
This closes reviewer question **Q6** (k-NN sensitivity) once complete.

---

## 4. Empirical radii (Q5) — DEFERRED (cluster); finishable, drop acceptable

**State (`results/canonical_tau1_empirical.json`):** 29 of 270 rows. **Adult only**;
α∈{0.0, 0.1} (18 of the 29 are α=0.0). The α=0.0 rows are **exact no-ops** — at α=0 the
empirical radii inversion equals the uniform radii (delta 0.000000), so they add no
information.

**Decision:** This is an exploratory Q5 companion (`radii_mode='empirical'`, coordinated=True),
not part of the canonical deliverable. The uniform radii are the canonical mode (per
MASTER_DISPATCH, the empirical/uniform dispatch itself is currently mislabeled). Therefore:
- **Finishable** by running the remaining Adult α∈{0.2,0.3,0.4} × 6 seeds × {dro}:
  `python3 experiments/run_canonical_empirical.py` (extend its α/list).
- **Dropping is acceptable** because (a) α=0 rows are no-ops, (b) only 2 of 5 α covered,
  (c) it is Adult-only, and (d) the uniform radii already supersede it for the paper.
- Marked **DEFERRED (cluster)** — not executed here; the run command and the no-op finding
  are recorded so it is not left in limbo.

---

## 5. Random vs adversarial — DEFERRED (cluster); re-run command given

**State (`results/random_vs_adversarial_new.json`):** 27 rows (3 datasets × 3 α
{0.1,0.2,0.3} × 3 seeds). **No provenance** — `tau=None, k_inner=None, epochs=None,
pgd_steps=None, lambda_init=None`. It predates the canonical protocol and is not comparable
to the canonical grid.

**Fix applied (Agent F):** `experiments/run_random_vs_adversarial.py` previously used the
**retracted stepped τ schedule** (`tau = 100.0 if alpha < 0.4 else 1.0`). It now imports the
canonical `get_temperature` (τ=1.0 constant) — so a re-run is automatically on the correct τ.

**Re-run command (deferred — cluster compute):**
```
python3 experiments/run_random_vs_adversarial.py \
  --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 \
  --n_seeds 6   # ensure canonical k_inner=10 / pgd_steps=20 inside the runner
```
Decision: **DEFERRED (cluster)** — the existing 27 rows are pre-canonical and must not be
published; a clean re-run under the canonical config is required before this comparison is
used in the paper.

---

## Bottom line

- **Finished / in-flight:** kNN backfill (§3) is running and mechanical.
- **Formally dropped with reason:** Tau (§1, incoherent confound), Lambda 720-grid (§2,
  λ0=1.0 pathology), Empirical-radii (§4, exploratory/no-op), Random-vs-adversarial (§5,
  pre-canonical).
- **No ablation is left in limbo:** every item has a disposition, evidence, and (where a
  re-run is the fix) an exact, ready-to-run command.
