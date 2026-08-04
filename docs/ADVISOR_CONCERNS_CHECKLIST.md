# Advisor concerns checklist (Manisha / Kuldeep / Ganesh chat)

**Source:** Google Chat May 19 – Jun 30 2026 (+ project work through Aug 4)  
**Verified against:** `results/canonical_tau1.json` (540), UTKFace 90 REAL, STATUS, paper/report  
**Updated:** 2026-08-04 (post Aug 10 final gate — aligned Claude + Grok sessions)

---

## A. Direct tasks (Manisha)

| # | Who / when | Ask | Current status | Evidence |
|---|------------|-----|----------------|----------|
| **M1** | Manisha May 19 | Email **supin.gopi** for **flair2** account | **DONE** access; flair2 **PARKED** (GPU live; torch not installed — Mac ran experiment) | STATUS §5, UTKFACE_STATUS |
| **M2** | Manisha May 19 | **PGD for fairness** (DP only, IF only, Combined) + DRO vs Naive | **DONE** FairnessTargetedPGD + full grid | `src/corruption/adversarial.py`, canonical 540 |
| **M3** | Manisha May 19 | **UTKFace**, same experiment | **DONE local REAL 90/90** (MPS ResNet18 features); flair2 optional not required | `utkface_features.npz`, `utkface_canonical.json`, paper §results-utkface |
| **M4** | Manisha Jun 2 | **Improve DP adversarial attack**, then **redo all experiments** | **DONE** gradient fairness PGD; redone as τ=1, k=10, n=6, 540 | MEETING, STATUS |
| **M5** | Manisha ongoing | Keep group updated / ping if stuck | Meeting handout shared; Aug 10 package ready | MEETING_HANDOUT |

---

## B. Kuldeep — May 29 questions

| # | Ask | Answer / status |
|---|-----|-----------------|
| **K-May29-a** | At α=0.1, DRO not significantly better — attack too weak? | Under **τ=1** fixed: Adult/Credit show DRO better on DP at α=0.1 (Adult DP **5/6**, p=0.031). |
| **K-May29-b** | Does the attack affect the **radius**? | Radii use α (+ empirical path if known attack). Canonical = **uniform** radii_mode. |
| **K-May29-c** | If attack too weak, should DRO perform well? | With τ=1, DRO is competitive/better at low α (see MEETING tables). |

---

## C. Your Q1–Q13 (Jun 9) + Kuldeep answers + our delivery

| Q | Topic | Kuldeep’s direction | Project status NOW |
|---|--------|---------------------|--------------------|
| **Q1** | “DRO fragile” valid? / bugs? | Try **λ_init / lr** tuning | **Narrative flipped:** fragility was **τ=100 artifact**. Fixed **τ=1** → DRO wins Adult/Credit α≤0.2. |
| **Q2** | Adult two-regime (worse α≤0.3, better 0.4) | Tied to τ schedule | **Resolved as artifact** of stepped τ=100→1. Under fixed τ=1, claim only defensible α≤0.2 on accuracy. |
| **Q3** | LSAC DP attack odd | **Dataset bias**; IF may be better for LSAC | **DONE honesty:** LSAC/DP **degenerate** (0/6, pinned ~0.90). LSAC/**Combined** genuine wins. |
| **Q4** | LSAC α=0 anomaly | Expected imbalance | Covered under LSAC degeneracy / constant predictor |
| **Q5** | Radii formula vs coordinated attack | **Empirical not theoretical** | Empirical radii path + appendix; **canonical uses uniform** |
| **Q6** | IF k-NN within-group | **Ablation k=5,10,15** | Historical ablation; not part of locked 540 claim |
| **Q7** | IF attack decreases DP | Coupling expected | **Confirmed:** IF metric vs DP under IF can diverge (Adult α=0.3: IF win, DP loss) |
| **Q8** | Scope of “redo all” | Tabular + attack; UTKFace separate | Tabular **540 complete**; UTKFace **90/90 REAL** |
| **Q9** | Seeds for Wilcoxon | n≥6 for p&lt;0.05 | **n=6 locked** on all 540 + UTKFace |
| **Q10** | K_inner=5 vs 10 | Paper says 10 | **k_inner=10** on all 540 |
| **Q11** | Absolute DP vs % | Prefer absolute | **Absolute DP** in tables/figures |
| **Q12** | τ=100 then τ=1 | **Fix τ for all α** in main | **τ=1.0 only** in canonical |
| **Q13** | UTKFace vs tabular priority | Finish tabular; UTKFace when ready | Both done: tabular 540 + UTKFace 90 REAL |

**Extra Kuldeep (Jun 9):** high-α framed vs **constant predictor**; hyperparam grid secondary to τ fix.

---

## D. Kuldeep live meeting asks (Jun 16–30)

| # | Ask | Status |
|---|-----|--------|
| Share **α=0.1 and 0.2** results | **DONE** — full n=6 in MEETING / canonical |
| **Accuracy plots** x=α, y=acc | **DONE** figures from 540 |
| Adult accuracy “must be ≥0.78” concern | At α≤0.2 competitive; **α≥0.3 below constant predictor (~0.75)** — stated honestly |
| Constant predictor baseline | **DONE** Adult 0.7521, Credit 0.7788, LSAC 0.9016 |
| Same plots for **IF violation** | IF metric fixed; IF-attack results real; **mixed** |
| **τ / λ** for better tradeoff at high α | **Cannot** beat constant predictor at α≥0.3; **defensible α≤0.2** |
| “Advantage grows with α” at high α | **Empty** if below constant predictor — we state that |
| **Jun 30: verify all claims** | **Agent V + VERIFICATION_REPORT LIVE** — 5/6, LSAC pinned, IF mixed |

---

## E. What the paper story is NOW (vs old chat narratives)

| Old (wrong / superseded) | Current (locked) |
|--------------------------|------------------|
| DRO fragile under PGD | **τ=100 artifact**; fixed **τ=1** → robust on Adult & Credit α≤0.2 |
| 3 seeds / K=5 / stepped τ | **6 seeds, K=10, τ=1** everywhere |
| IF ≈ 0 / mislabelled | Cosine IF; full **180** IF-attack rows; **mixed** story |
| UTKFace synthetic / blocked | **REAL 90/90**; paper: **mixed clean-test pilot** (not Adult copy) |
| “6/6 every α” | **α=0.1 is 5/6** (must say) |

---

## F. Still open (optional; not Aug 10 blockers)

| Item | Notes |
|------|--------|
| flair2 full GPU / pixel-level | Parked; Mac already has REAL feature pilot |
| Empirical-radii full table in paper | Q5 appendix; uniform is SSOT for 540; Wave-1 A5 → `empirical_radii.json` when done |
| Extra constant-predictor fig polish | Prefer figD1–D4 from 540 if re-sharing |
| kNN / λ grid / random-vs-adv / τ / K_inner | Wave-1 drivers live (`run_a*.py`); sequential on Mac; separate JSONs only |
| IF@α=0.3 claim | ✅ Formalized (`results/if_wilcoxon_n4_summary.md` + paper Q7) |

---

## G. One-pager if asked

1. **Manisha:** PGD fairness attacks done; UTKFace **REAL 90/90** (local MPS; flair2 proven & parked).  
2. **Kuldeep τ:** fixed to 1 — old “fragile DRO” was temperature.  
3. **Protocol:** n=6, K_inner=10, **540 complete**.  
4. **Honest:** Adult DP α=0.1 is 5/6; α≥0.3 below constant predictor on Adult/Credit; LSAC/DP degenerate; IF mixed; UTKFace mixed pilot.  
5. **Verify:** every number rechecked against `canonical_tau1.json` / `utkface_summary.md`.

**Files:** `docs/MEETING_HANDOUT_2026-08-04.md`, `docs/VERIFICATION_REPORT.md`, `docs/KULDEEP_CORRECTION.md`, `docs/AUG10_SUBMISSION_CHECKLIST.md`, `figures/fig_tau1_headline.pdf`, `figures/fig_final_wilcoxon_table.pdf`.
