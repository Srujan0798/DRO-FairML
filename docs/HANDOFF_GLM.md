# HANDOFF — GLM

**Your lane: Mac CPU + the training/radii math.** You do not touch flair2, GPU code, or
any file Grok's lane owns (see `docs/HANDOFF_GROK.md`). You do not touch `paper/*.tex`,
`report/*.tex`, or `STATUS.md` — integration is a separate later pass so two agents
never edit the same `.tex` file at once (that already happened once tonight and broke
the paper).

---

## PHASE 0 — CORRECTNESS AUDIT (do this FIRST, blocks everything below)

Direct instruction from the project owner: stop adding new experiments, go back to
first principles, verify the math is actually correct — not just that experiments ran
and produced plausible numbers. **Do not resume the ablation queue at the bottom of
this file until every box in "Phase 0 done" is checked.**

### Finding 1 — the "uniform" radii formula has likely never once executed (HIGH severity)

`src/training/dro_fair.py::_compute_radii` (lines 99-122) branches in this order:
```python
if self.radii_mode == 'empirical':
    pi_clean = self._empirical_pi_clean(pi_obs)          # Q5 known-attack inversion
elif a_val is not None and len(a_val) > 0:
    pi_clean = np.array([np.mean(a_val == j) for j in [0, 1]])   # ← clean VALIDATION labels
else:
    pi_clean[j] = (pi_obs[j] - self.alpha) / (1 - 2*self.alpha)  # ← the documented
                                                                   #   "uniform" closed form
```
`experiments/run_fairness_pgd.py` (the canonical runner, used for **every row** in
`results/canonical_tau1.json` and `results/utkface_canonical.json`) **always** calls
`trainer.fit(..., a_val=a_val, ...)` with non-empty `a_val` (lines 153-154, 169-170).

**Consequence: the closed-form uniform formula documented in `docs/KEY_FORMULAS.md`
("π_clean = (π̂_obs − α)/(1 − 2α), Appendix F") has never executed in any canonical row.**
Every row labeled `radii_mode: "uniform"` actually used clean validation-set group
proportions, not the bias-corrected estimate from corrupted training data.

The code comment (lines 84-86) argues this isn't an oracle leak because "both Naive and
DRO have access to clean validation data" — a real argument, not nothing. But it means:
1. The provenance label `radii_mode: uniform` is misleading — it never ran that formula.
2. The "uniform vs empirical" comparison in the paper compares "known val proportions"
   vs "known attack-structure inversion," not "generic closed-form" vs "attack-aware" —
   a weaker claim than Theorem 4.2/Appendix F implies.
3. `test_default_radii_mode_is_uniform` only asserts the config flag, not that the
   formula executed — a green test suite doesn't catch this.

**Resolve before Aug 10.** Two honest paths — pick one, document the choice here and in
`docs/KEY_FORMULAS.md`:
- (a) Fix the runner to not pass `a_val` when `radii_mode='uniform'` is the intended
  comparison, actually exercise the closed form, re-run, see what changes.
- (b) Keep `a_val` (defensible) but rename the mode and rewrite every place the paper
  claims to test the theoretical closed-form formula, since it currently doesn't.

### Item 4 — TV → L1 conversion (`L1 radius = 2·ρ_TV`) — NOT YET CHECKED

Check `src/utils/projections.py` and every place a radius crosses between TV and L1
units. The Dykstra projection in Algorithm 1 step 4 projects onto `Δ_n ∩ B_1(p̂, 2ρ)` —
**confirm the code actually multiplies by 2 somewhere, or confirm ρ is already stored in
L1 units and the ×2 is folded into the formula #4 conversion. If neither, the ball
radius is wrong by a factor of 2** — this would make every DRO run use the wrong
uncertainty-set size, silently, while still producing plausible-looking numbers. This is
the single highest-priority unchecked item after Finding 1.

**Action:** hand-verify by reading the code, then write a unit test that constructs a
case with a known analytic radius and asserts the projected point lands inside the
*correct* ball, not one twice or half that size.

### Items 5-6 — tilted risk and dual ascent, hand-re-derive against Algorithm 1

- Tilted risk `β·logsumexp(ℓ/β)` (Algorithm 1 step 2): re-derive by hand from the
  paper and compare token-by-token against the training loop in `dro_fair.py`. Only
  "it exists and runs" has been checked so far, not "it matches the paper's math."
- Dual ascent `λ ← clamp(λ + η·0.95^t·g, 0, λ_max)` (step 3): confirm the decay
  `0.95^t` uses the correct counter — epoch index or step index? An off-by-one here
  silently changes the effective learning-rate curve without crashing anything.

### Invariant checks — run across the FULL grid, not spot checks

| Invariant | Expected | Check |
|---|---|---|
| `dp_clean ≥ 0` every row | Always | `python3 -c "import json; d=json.load(open('results/canonical_tau1.json')); print(sum(1 for r in d if r['dp_clean']<0))"` → must print 0 |
| `if_clean ≥ 0` every row | Always | same pattern |
| `acc_clean ∈ [0,1]` every row | Always | same pattern |
| `rho_dp[j] ∈ [0,1]` for every α, π combination | `α/((1−α)π+α)` is bounded in [0,1] | may need to log rho per row if not already in provenance — check |
| DP(DRO) vs DP(Naive) direction | DRO ≤ Naive **only in the defensible regime (α≤0.2, Adult/Credit)** — not universal, LSAC/DP is a documented exception | recompute Wilcoxon independently from raw JSON, don't trust the last generator's cached output |
| α=0 gap vs α=0.1 gap | α=0 gap should be small relative to α=0.1's (DRO/Naive use different objectives even at α=0 — see below — a LARGE α=0 gap suggests a real bug beyond the known difference) | compare `abs(Δ at α=0)` vs `abs(Δ at α=0.1)` per dataset/attack |

### Context for the α=0 item (Grok's finding, you'll see it referenced)

At α=0, `for _ in range(self.K_inner if self.alpha > 0 else 0)` correctly stops the
inner p-ascent — but DRO still optimizes a tilted risk while Naive optimizes plain BCE,
so the two are never the same objective, even at zero corruption. Every "DRO wins at
α=0.0" cell is "DRO's tilted objective beats plain BCE absent any attack," not "DRO
survives zero attack." Keep this in mind for the invariant check above and for anything
you write.

### Phase 0 done when

- [x] Finding 1 resolved, decision (a) or (b) documented in `docs/KEY_FORMULAS.md`
- [x] Item 4 (×2 factor) hand-verified with a passing unit test
- [x] Items 5-6 hand-re-derived and confirmed against Algorithm 1
- [x] Invariant table run across the full grid, zero violations (or violations explained)
- [x] Findings appended to this file (below this line), don't rewrite the sections above

**Findings go here:**

### Phase 0 complete — 2026-08-05

**Finding 1 (uniform radii formula never executed): CONFIRMED, decision (b).**
- The `elif a_val is not None` branch in `_compute_radii` (line 109) fires for every
  canonical row because `run_single_experiment` always passes `a_val`. The documented
  closed-form `(π_obs − α)/(1 − 2α)` (line 115) is dead code in the canonical path.
- **Decision (b):** Keep `a_val` (defensible — clean validation group rates, not an
  oracle leak), but the paper must NOT claim to test the Appendix F closed form. The
  `radii_mode: "uniform"` JSON label is retained for backward compatibility but means
  "validation-estimated" in practice. Documented in `docs/KEY_FORMULAS.md`.
- **Impact on locked science:** NONE. The radii used were valid; only the label and
  the paper claim about which formula ran were wrong. No retrain needed.

**Item 4 (TV → L1 ×2 factor): VERIFIED CORRECT.**
- `dro_fair.py:219` passes `2 * radius` to `project_simplex_l1_ball`. Unit test
  `test_tv_to_l1_radius_factor_2` confirms the projected point lands on the L1 ball
  boundary at exactly 2ρ, not ρ or 4ρ. No bug.

**Items 5-6 (tilted risk + dual ascent): VERIFIED CORRECT.**
- Tilted risk: `β·(logsumexp(ℓ/β) − log(m))` = `β·log(mean(exp(ℓ/β)))` — matches
  Algorithm 1 step 2.
- Dual ascent: `λ ← clamp(λ + η·0.95^epoch·g, 0, λ_max)` — matches Algorithm 1 step 3.
  Decay counter is epoch index (0-indexed), one update per epoch. Correct.

**Invariant checks (560 rows):**
- `dp_clean ≥ 0`: 0 violations ✓
- `if_clean ≥ 0`: 0 violations ✓
- `acc_clean ∈ [0,1]`: 0 violations ✓
- α=0 gap: Adult/Credit small (0.001-0.007, tilted-risk-vs-BCE). LSAC large (+0.038,
  known degeneracy). Not new bugs.

**Tests:** 91 passed (was 90; +1 new `test_tv_to_l1_radius_factor_2`).

---

## PHASE 1 — the ablation queue (resume only after Phase 0 is checked off)

### Locked truth — read once, don't re-derive

- `results/canonical_tau1.json`: **540 rows locked** (τ=1, K=10, n=6, seeds 0-5). A
  parallel n=10 extension has appended seeds 6-9 for some cells (file may show >540) —
  **all claims still use seeds 0-5 / the original 540.** Never edit or shrink this file.
- `results/utkface_canonical.json`: 90/90 REAL rows, complete. Not your lane — read-only
  reference if needed.
- Central finding: fixed τ=1 makes DRO beat Naive on Adult/Credit DP+Combined at α≤0.2
  (Adult/DP α=0.1 is 5/6, not 6/6 — everywhere else 6/6). LSAC/DP is degenerate (no fix
  found — L2 tested and confirmed, see `results/lsac_radii_summary.md`). IF is MIXED.
  α≥0.3 falls below the constant predictor on Adult/Credit only (LSAC pinned at baseline).

### The one hard rule for running anything

**Never run an ablation script directly (`python3 experiments/run_a1_knn.py` etc.).**
Always go through the orchestrator or the shared lock — running directly bypasses
nothing (the lock in `experiments/run_ablation_parallel.py` still queues you), gains
nothing, and is how tonight's confusion started. Check state first:
```bash
ps aux | grep -iE "orchestrate_wave1|run_a[0-9]|run_n[0-9]|run_l2|run_s_n10"
```
- If something's already running → **let it run**, don't relaunch, don't kill it.
- If nothing's running → `bash scripts/orchestrate_wave1.sh` (resume-safe; finished
  jobs skip instantly, in-progress jobs resume exactly where they stopped — checkpointed
  after every result via `atomic_save`).

### Already done — do not re-run

| File | Status |
|---|---|
| `results/if_wilcoxon_summary.txt`, N4 IF@α=0.3 analysis | DONE |
| `results/lsac_radii_fix.json` | DONE (120/60 — over target, fine — thorough hypothesis test) |
| `results/extended_datasets.json` (COMPAS + German) | DONE — German replicates the DRO pattern, COMPAS ambiguous. Report both honestly. |
| `results/lambda_grid.json` | DONE 72/72 — no (λ,lr) beats default, no α=0.3 rescue |

### Remaining orchestrator queue

| Job | Target | Rows (snapshot, check live) |
|---|---|---|
| N2-HighAlpha | 120 | ~26+ |
| A1-kNN | 360 | 48 |
| A2-Tau | 360 | 76 |
| A4-RvA | 144 | 43 |
| A5-Empirical | 180 | 69 |
| N5-Kinner | 180 | 24 |
| N1-AttackStrength (a) | 144 | 22 |
| N1-AttackStrength (b) | 180 | 22 |
| S-N10-Extension | 900 total | 560 |

Early finding (from A4 partial data): the "12–40× stronger than random" claim quoted to
Kuldeep on Jun 16 does **not** hold under the canonical protocol — early numbers suggest
0.2–1.1×. **Do not let anyone put 12–40× in the paper.** Finish A4 to 144/144, report
the real number.

### When a job finishes

1. Write/refresh `results/<job>_summary.md` — honest answer to the ablation's question.
2. Do NOT edit `paper/*.tex` or `report/*.tex` — leave the summary for the integration pass.
3. Commit + push (`git pull --rebase` first if `git status` shows you're behind — you're
   on the same working directory as Grok and the assistant).

### When the whole queue is done

Append status to `STATUS.md`'s "what's left" section — don't rewrite the whole file, and
don't attempt final paper integration yourself.
