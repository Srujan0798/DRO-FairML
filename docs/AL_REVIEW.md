# Independent Adversarial Review of the DRO-FAIR-AL Claim (TASK E)

**Reviewer:** fresh, no project history. **Date:** 2026-08-07.
**Reviewed commit:** `53e4716` (HEAD of `al-e-review` worktree).
**Method:** recomputation from raw JSON + code only; existing `results/*_summary.md`
and the paper/report text were treated as *claims under review*, never as ground truth.
All numbers below were recomputed from `results/*.json` and live training runs.

**Verdict in one line:** the DRO-FAIR-AL claim survives adversarial review on its
core quantitative assertions (DP reduction, Wilcoxon p, lambda-starvation diagnosis,
µ=0 no-op, no leakage), but the prose overstates the accuracy side ("held or
improved" is false at α=0.0 and hides a full seed-3 collapse at α=0.2), and several
verdicts (Credit, the radius compound CONFLICT) hinge on an arbitrarily chosen
`+0.005` accuracy margin that was pre-registered but never independently justified.

---

## A. Items CONFIRMED as correct (with evidence)

### A1. Step 1 — the augmented-Lagrangian gradient is correctly implemented.
`src/training/dro_fair.py:356-363` adds `0.5 * mu * g_dp * g_dp` (and `g_if`)
to the scalar `total_loss` that is `.backward()`ed. Hand check:
`d/dθ[(µ/2)·g²] = µ·g·(dg/dθ)` is exactly what autograd produces — the term is
added to the correct scalar, with a positive sign (minimization of the penalty),
once per active constraint (no double counting). Numerical verification
(`experiments/verify_al_gradient.py`): for µ=20, `grad(total_loss)` equals
`grad(L_tilt) + µ·g_dp·grad(g_dp) + µ·g_if·grad(g_if)` to relative error <2e-7
(PASS); µ=0 gradient is bit-identical to the base-loss gradient (PASS).
`g_dp` is an absolute value (`dro_fair.py:271`) and `g_if` a ReLU-weighted sum
(`dro_fair.py:283-284`), so both are non-negative by construction — no `max(g,0)`
clamp is needed and none is missing. **The "no sign error / wrong tensor /
double count" class of bug does not exist.**

### A2. Step 2 — µ=0 is a true end-to-end no-op (independently verified).
Git shows `4b4743a` "DRO-FAIR-AL: augmented-Lagrangian constraint penalty" is the
introducing commit; its parent is `4adb128`. Ran the SAME single config
(Adult, α=0.2, seed=0, DP attack, DRO, epochs=60, K_inner=10, pgd_steps=20,
tau=1.0, n_seeds_planned=6) at `4adb128` and at HEAD with `aug_lagrangian_mu=0.0`
(scratch worktree `/var/folders/.../opencode/al-parent`, script
`experiments/run_noop_check.py`):

| metric | parent `4adb128` | HEAD µ=0.0 | match |
|---|---|---|---|
| acc_clean | 0.7547816473189608 | 0.7547816473189608 | exact |
| dp_clean  | 0.2459459751844406 | 0.2459459751844406 | exact |
| if_clean  | 0.04569387063384056 | 0.04569387063384056 | exact |

Full float precision. The unit test (`tests/test_aug_lagrangian.py`) was NOT relied
upon; this was run end-to-end. Sanity: µ=20 on the same cell gives
acc 0.7751243781094528, dp 0.07365065813064575, if 0.00229227589443326 —
byte-identical to the stored `results/mu_sensitivity.json` row, proving the µ
parameter is genuinely threaded through and µ=20 does change training.

### A3. Step 3 — headline Wilcoxon recomputed, p = 0.015625 (matches 0.0156).
Adult, α=0.2, DP attack. From `results/canonical_tau1.json`
(tau==1.0, method=='dro', attack=='dp', alpha==0.2, seed<6) and
`results/mu_sensitivity.json` (dataset=='adult', alpha==0.2, aug_lagrangian_mu==20.0):

| seed | canonical dp_clean | AL(µ=20) dp_clean | diff | % reduction |
|---|---|---|---|---|
| 0 | 0.24594584 | 0.07365066 | +0.17230 | 70.1% |
| 1 | 0.23443776 | 0.08720300 | +0.14723 | 62.8% |
| 2 | 0.23099940 | 0.08520582 | +0.14579 | 63.1% |
| 3 | 0.21793075 | 0.00721014 | +0.21072 | 96.7% |
| 4 | 0.24212202 | 0.08513087 | +0.15699 | 64.8% |
| 5 | 0.22885777 | 0.07059854 | +0.15826 | 69.2% |

Both arrays are sorted by seed → **pairing is by SEED** (the historical bug class in
`tests/test_wilcoxon_seed_pairing.py` is not present here; unpaired-order handling
was not needed and the seed keys were verified identical). `scipy.stats.wilcoxon(
dro, al, alternative='greater')` → **p = 0.015625**, two-sided 0.03125. All six
differences are positive → 6/6 seeds improve DP. Mean reduction 70.8% (α=0.2);
81.7% at α=0.0 (same pairing, p=0.015625). Matches the claimed "70.8–81.7%,
p=0.0156, 6/6".

### A4. Step 4 — constant-predictor floors.
Computed via `src/data/datasets.py::get_dataset(..., random_state=42)` on the
TRAINING labels: Adult 0.752152, Credit 0.778821, LSAC 0.901885. Claimed floors
Adult 0.7521 ✓, Credit 0.7788 ✓, **LSAC 0.9016 ✗ (training majority is 0.9019)**.
The 0.9016 value equals the LSAC *test* majority fraction (0.901578) — see defect
D4. Note the AL-collapse accuracy on LSAC (0.9016, `aug_lagrangian_extended.json`)
is exactly the test majority rate, confirming the "constant predictor" there is the
test rate.

### A5. Step 5 — radius-compound numbers verified; qualitative verdict robust to margin.
From `results/al_radius_compound.json` (adult, α=0.2, radii_scale=2.0, µ=20, 6 seeds):

| seed | acc_clean |
|---|---|
| 0 | 0.759757 |
| 1 | 0.757877 |
| 2 | 0.756882 |
| 3 | 0.752128 |
| 4 | 0.757214 |
| 5 | 0.752902 |

Mean acc = **0.756127** ✓ (claim 0.756127); mean dp = 0.0139, R = 94.0% vs canonical ✓.
Threshold floor+0.005 = 0.7571: **mean 0.756127 < 0.7571 ✓**. Margin sensitivity:
margin 0.01 → threshold 0.7622, all 6 seeds below (verdict stronger); margin 0.0 →
threshold 0.7522, mean 0.7561 ABOVE floor (mean-based verdict flips), 1/6 seeds
(seed 3, 0.752128) below the exact floor. Note also the combined arm's mean accuracy
(0.7561) is *below* the canonical DRO mean (0.7586), so "collapse" is qualitatively
right even without the margin. But the specific claim "3 of 6 seeds at or below the
**floor**" is wrong — see defect D3.

### A6. Step 7 — no leakage; the only code-path difference is the penalty term.
Read `experiments/run_fairness_pgd.py::run_single_experiment` end to end
(lines 47-222). Both µ=0 and µ>0 share: identical RNG seeding
(`random.seed/np.random.seed/torch.manual_seed(seed)`, lines 86-90), identical
`get_dataset(dataset_name, random_state=seed)` split, identical attack object and
`corrupt()` call, identical trainer construction and `fit()`, identical
`compute_metrics_torch` evaluation on test. The single µ-dependent argument flows
into `DroFairTrainer(aug_lagrangian_mu=...)` and only alters the guarded
`if self.aug_lagrangian_mu > 0:` branch in `fit()` (`dro_fair.py:356-363`), which
is skipped entirely at µ=0 (no extra autograd nodes). No extra pass over validation
data, no different split, no different seeding order. `X_val` is used only for
per-epoch validation metrics (no gradient), identically in both cases.

### A7. Supporting diagnosis — lambda starvation is real and correctly quantified.
From `results/history_adult_dp_0.2_0_dro_canonical.json` (the exact Adult α=0.2
seed-0 canonical history): max λ_dp over 60 epochs = **0.0119** (claimed ≤0.012,
ceiling 1.5, i.e. 126× below); peak fairness penalty λ·g = **0.00293** at epoch 57,
where train_loss = 0.5405 → **0.54% of the loss** (claimed ~0.5%, "0.538").
Diagnosis CONFIRMED from raw history.

### A8. Step 8 — canonical baseline itself is near-degenerate; exclusion changes nothing.
Adult α=0.2 canonical DRO accuracies vs exact floor 0.752152:
seed 5 = **0.748811, BELOW the floor** (itself degenerate); seeds 0/1/2 within
+0.0034 of the floor (borderline); only seeds 3/4 clear. Excluding seed 5 from the
Wilcoxon: R = 71.1%, p = 0.03125 — still significant. The paper's conclusions do
not depend on the degenerate canonical seed.

### A9. Boundary claims spot-checked (all hold).
- α=0.4: no µ in {0.5,1,2,5,10,20} safe — worst accs 0.5398/0.4983/0.4460/0.3495
  (µ=5)/0.2742 (µ=20), all far below the floor; the −80.7% figure is the µ=5 cell
  (acc 0.3495). ✓ (Slight wording imprecision: µ=20 is "most aggressive" but gives
  −94.0%; −80.7% is µ=5.)
- Credit not rescued: µ∈{0.5,1,2} accs 0.7808/0.7780/0.7784, all below the
  pre-registered SAFE bound floor+0.005 = 0.7838. ✓ (mechanically per pre-reg rule;
  see D5 for the margin caveat).
- Combined attack α=0.2 µ=5: −46.8%, p=0.0156, acc 0.7599→0.8032 ✓.
- IF attack α=0.2 µ=5: acc 0.7522 ≈ floor, borderline as claimed ✓.
- LSAC collapses under AL to exactly 0.9016 (test majority rate) ✓.
- The canonical grid's IF-column staleness (≈1e-11) for DP/Combined rows is real
  and I reproduced it (see D6) — but the paper discloses it in
  `paper/sections/discussion.tex:106-120` with exactly the claims I verified:
  accuracy reproduces exactly and DP shifts by O(10⁻⁷) under the corrected metric.

---

## B. DEFECTS FOUND (severity, evidence)

### D1 — MODERATE. "Accuracy held or improved / Pareto improvement" is false at α=0.0.
Claim: "reduces DP by 70.8–81.7% ... while holding or improving accuracy"
(task text); "accuracy held or improved --- a Pareto improvement, not a
fairness/accuracy trade" (`paper/sections/results.tex:410-411`). Recomputed from
raw rows: at α=0.0, AL(µ=20) mean accuracy is **0.7966 vs canonical 0.8147
(−0.0181), and all 6 seeds lose accuracy** (per-seed −0.014 to −0.022). The table
prints the drop (0.8147→0.7966), so it is transparent, but the prose "held or
improved / Pareto improvement" is inaccurate for α=0.0. At α=0.2 the aggregate
improves (0.7586→0.7783) but only 5/6 seeds improve (see D2). A fair wording is
"DP reduction at a small accuracy cost at α=0.0; DP reduction with modest aggregate
accuracy gain at α=0.2".

### D2 — MODERATE. Seed 3 collapses to the exact constant predictor at µ=20, hidden by the aggregate.
At α=0.2, µ=20, seed 3: acc_clean = **0.7521282434463501 = the test majority rate
exactly** (computed independently from `get_dataset('adult', random_state=3)` test
labels) — i.e. the model emits constant-negative predictions on the test set for
that seed (canonical DRO at that seed: 0.7780). So 1/6 seeds at the recommended µ
is a full constant-predictor collapse; seeds 5 (0.757103, +0.005 over the majority
rate) and 0 (per TASK B: predicted-positive rates 0.0%/4.4%) are near-collapse.
The paper's aggregate "accuracy held or improved, 6/6" (attached to DP) plus the
mean 0.7783 hides that the mechanism is partly "suppress positive predictions
toward the majority class" (see section C for the direct per-seed confirmation).

### D3 — MINOR. "3 of the 6 seeds are at or below the floor itself" is inaccurate.
Recomputed with the exact floor 0.7521518: only **1/6 seeds (seed 3, acc 0.752128)
is at or below the floor**; with the rounded 0.7521, 0/6 (seed 3 is 0.752128 >
0.7521). The correct statement is **3/6 seeds (2,3,5) at or below the DEGEN
threshold 0.7571 (= floor + 0.005)**, which is what the pre-reg and
`summarize_al_radius_compound.py` actually use. The verdict (CONFLICT) is
unchanged, but the phrasing in the task description conflates floor and threshold.

### D4 — MINOR. LSAC constant-predictor baseline 0.9016 is the TEST majority, not training (0.9019).
The paper/report pin "LSAC ≈ 0.9016"; `experiments/loaders.py:30-34` hardcodes a
0.9016 fallback while its own computed value `constant_predictor_acc('lsac')`
returns **0.901885 (training majority)**. Only Adult and Credit floors match the
training majority (0.7522/0.7788). The test-split majority (0.901578) is arguably
the correct threshold for test accuracy, so the number is defensible — but the
provenance is unstated and the code/report disagree with each other at the third
decimal.

### D5 — MINOR. The +0.005 degeneracy/safety margin is arbitrary and several verdicts hinge on it.
The SAFE/DEGEN rule `acc > floor + 0.005` is used for: Credit rescue (µ=0.5 gives
acc 0.7808 > floor 0.7788 but < floor+0.005 → ruled "not safe"), the α=0.2 Adult
safety frontier, and the radius-compound CONFLICT. It was fixed in the
pre-registrations before data existed (methodologically sound), but with a 0.0
margin the Credit µ=0.5 cell flips to safe and the compound-mean verdict flips
(mean 0.7561 > floor 0.7522). The margin's magnitude has no independent
justification, and the paper's α=0.2 regime is itself only +0.0065 over the floor
(canonical mean 0.7586), so several "safe"/"unsafe" classifications sit on the
knife's edge of this arbitrary constant.

### D6 — MINOR (already disclosed, verified). canonical_tau1.json seeds 0–5 IF column is stale.
Seeds 0–5 of the locked canonical file record `if_clean ≈ 1e-11` (floating-point
dust from the pre-cosine-fix Euclidean IF graph); seeds 6–9 record real values
(≈0.047–0.051). My full rerun of the seed-0 cell under current code gives
if_clean = 0.0457 and dp_clean = 0.2459459751844406 vs stored 0.24594584107398987
(Δ 1.3e-7) with acc identical. The paper discloses this exactly
(`paper/sections/discussion.tex:106-120`, "reproduces accuracy exactly and shifts
DP by O(10⁻⁷)") and my recomputation confirms that claim, so this is *not* a defect
in the DP-based DRO-FAIR-AL claim — listed for completeness. Anyone reading
if_clean from the seeds 0–5 canonical rows must not treat it as a measurement.

---

## C. Step 6 — seed-dependence of the µ=20 predicted-positive-rate collapse

Direct per-group predicted-positive-rate training runs were run for seeds 0–3 ×
{µ=0, µ=20} (`experiments/run_al_mechanism_seed_scan.py`, output
`results/al_mechanism_seed_scan.json`), replicating the `run_al_mechanism.py`
pattern. Predicted-positive rate per protected group on the attacked TRAINING set:

| seed | canonical µ=0 (g0 / g1) | µ=20 (g0 / g1) | verdict at µ=20 |
|---|---|---|---|
| 0 | 0.164 / 0.539 | **0.000 / 0.044** | collapse |
| 1 | 0.158 / 0.531 | 0.000 / 0.149 | partial collapse (g1 retains 15%) |
| 2 | 0.185 / 0.501 | **0.000 / 0.054** | collapse |
| 3 | 0.151 / 0.472 | **0.000 / 0.000** | FULL constant-negative predictor |

Seed 3 at µ=20 predicts positive for *no sample of either group* on train AND test
(pos rates 0.000/0.000; test accuracy 0.752128 = the test majority rate to 16
digits). This confirms the stored-data signal: the predicted-positive-rate
collapse at µ=20 is **general, not seed-0-specific** — seeds 0, 2, 3 fully or
nearly collapse, seed 1 partially, and only seeds 4–5 (stored accs 0.7945/0.7571)
retain more signal. At canonical µ=0 every seed keeps healthy positive rates
(g0 15–19%, g1 47–54%).

**Conclusion for the mechanism debate:** "AL denoises the attack" is too benign a
description of µ=20 — **"AL pushes toward the majority (negative) class"** is the
more accurate mechanism, with a full constant-predictor collapse on at least one
seed (3) of the six. The aggregate safety margin (mean 0.7783) is thinner
per-seed than the mean suggests; this is a caveat on the µ=20 recommendation, not
a refutation of the DP-reduction result (which is real and reproducible).

---

## D. Deliverable checks
- `pytest tests/ -q` from repo root: **101 passed, 1 warning** (unknown `slow`
  mark in `tests/conftest.py:28` — infra nit, not a failure). Green.
- No merge performed; branch left for review.

## E. Bottom line
The core DRO-FAIR-AL quantitative claims — DP reduction 70.8–81.7%, seed-paired
Wilcoxon p=0.0156, µ=0 no-op, lambda-starvation diagnosis, α≤0.2 Adult-only scope
with α=0.4 and Credit boundaries — are **correct and reproducible**. The
"accuracy held or improved / Pareto" framing is overstated (D1, D2), a per-seed
collapse at the recommended setting is hidden by the mean (D2, C), and several
verdicts rest on an arbitrary pre-registered +0.005 margin (D5). These are
presentation/interpretation defects, not fabrication: every number in the tables
reproduced exactly from raw data.
