# Advisor concerns checklist (Manisha / Kuldeep / Ganesh chat)

**Source:** Google Chat May 19 – Jun 30 2026 (+ later project work to Aug 4)  
**Verified against:** `results/canonical_tau1.json` (540), STATUS, MEETING brief, code  
**Updated:** 2026-08-04

---

## A. Direct tasks (Manisha)

| # | Who / when | Ask | Current status | Evidence |
|---|------------|-----|----------------|----------|
| **M1** | Manisha May 19 | Email **supin.gopi** for **flair2** account | **DONE infrastructure** (access later granted; currently **PARKED**). Torch install stopped (slow wifi). | STATUS §5, UTKFACE_STATUS |
| **M2** | Manisha May 19 | **PGD for fairness** (DP only, IF only, Combined) + DRO vs Naive on Adult etc. | **DONE** FairnessTargetedPGD + full grid | `src/corruption/adversarial.py`, canonical 540 |
| **M3** | Manisha May 19 | **UTKFace on server**, same experiment | **PARTIAL** — REAL features + local MPS grid ~54/90; flair2 staged not required | `utkface_features.npz`, `utkface_canonical.json` |
| **M4** | Manisha Jun 2 | **Improve DP adversarial attack**, then **redo all experiments** | **DONE** gradient fairness PGD; redone as τ=1, k=10, n=6, 540 | MEETING, STATUS |
| **M5** | Manisha ongoing | Keep group updated / ping if stuck | Process | — |

---

## B. Kuldeep — May 29 questions

| # | Ask | Answer / status |
|---|-----|-----------------|
| **K-May29-a** | At α=0.1, DRO not significantly better — attack too weak? | Under **τ=1** fixed: Adult/Credit still show DRO better on DP at α=0.1 (Adult DP **5/6**, p=0.031). Weak-attack regime no longer “DRO loses.” |
| **K-May29-b** | Does the attack affect the **radius**? | Radii use α (+ empirical path if known attack). Documented; uniform mode in locked canonical. |
| **K-May29-c** | If attack too weak, should DRO perform well? | Yes in theory; with τ=1, DRO is competitive/better at low α (see MEETING tables). |

---

## C. Your Q1–Q13 (Jun 9) + Kuldeep answers + our delivery

| Q | Topic | Kuldeep’s direction | Project status NOW |
|---|--------|---------------------|--------------------|
| **Q1** | “DRO fragile” valid? / bugs? | Try **λ_init / lr** tuning; acc↓+DP↓ can fit | **Narrative flipped:** fragility was **τ=100 artifact**. Fixed **τ=1** → DRO wins Adult/Credit α≤0.2. Lambda grid was run (ablation; not the main claim). |
| **Q2** | Adult two-regime (worse α≤0.3, better 0.4) | Tied to τ schedule | **Resolved as artifact** of stepped τ=100→1. Under fixed τ=1, DRO better on DP across α (claim only defensible α≤0.2 on accuracy). |
| **Q3** | LSAC DP attack odd | **Dataset bias**; IF may be better for LSAC | **DONE honesty:** LSAC/DP **degenerate** (0/6, pinned ~0.90). LSAC/**Combined** genuine wins. Real IF-attack also fails LSAC on DP. |
| **Q4** | LSAC α=0 anomaly | Expected imbalance issues | Covered under LSAC degeneracy / constant predictor |
| **Q5** | Radii formula vs coordinated attack | **Empirical not theoretical** — if attack known, approximate per paper | Empirical radii path + appendix material; **canonical uses uniform** radii_mode with provenance |
| **Q6** | IF k-NN within-group | **Ablation k=5,10,15** | Ablation run historically; not part of locked 540 claim |
| **Q7** | IF attack decreases DP | Coupling expected | **Confirmed in full IF third:** IF metric vs DP under IF attack can diverge (Adult α=0.3: IF win, DP loss) |
| **Q8** | Scope of “redo all” | Tabular + attack; UTKFace separate | Tabular **540 complete**; UTKFace REAL in progress |
| **Q9** | Seeds for Wilcoxon | n≥6 for p&lt;0.05 | **n=6 locked** on all 540 |
| **Q10** | K_inner=5 vs 10 | Paper says 10 | **k_inner=10** on all 540 rows |
| **Q11** | Absolute DP vs % | Prefer absolute | **Absolute DP** in tables/figures |
| **Q12** | τ=100 then τ=1 | **Fix τ for all α** in main; other τ only ablation | **τ=1.0 only** in canonical |
| **Q13** | UTKFace vs tabular priority | Finish tabular; UTKFace when ready | Tabular done; UTKFace REAL ~54/90 local |

**Extra Kuldeep (Jun 9):**  
- If accuracy drops and DP drops, that can fit setup → high-α regime later framed vs **constant predictor**.  
- Hyperparam grid: start **Adult**, then discuss (Jun 13).

---

## D. Kuldeep live meeting asks (Jun 16–30) — plot / accuracy discipline

| # | Ask | Status |
|---|-----|--------|
| Share **α=0.1 and 0.2** results | **DONE** — full n=6 in MEETING / canonical |
| **Accuracy plots** x=α, y=acc | **DONE** historically; regenerate if sharing again from 540 |
| Adult accuracy “must be ≥0.78” concern | At α≤0.2 DRO stays competitive; **α≥0.3 falls below constant predictor (~0.75)** — we must say this |
| Constant predictor baseline | **DONE** Adult 0.7521, Credit 0.7788, LSAC 0.9016 |
| Same plots for **IF violation** | IF metric fixed; IF-attack results real; mixed |
| **τ / λ** for better tradeoff than constant predictor at high α | Tested: **cannot** beat constant predictor at α≥0.3 by τ/λ; **defensible α≤0.2** |
| Why skip α≥0.2 on some plots | Fixed (y-axis floor bug); full α range in honest plots |
| “Advantage grows with α” at high α | **Empty** if below constant predictor — we state that |
| **Jun 30: verify all claims** (AI overclaims) | **Agent V + VERIFICATION_REPORT LIVE** — Adult/DP **5/6** at 0.1; LSAC not “below”; wrong Naive acc fixed |

---

## E. What the paper story is NOW (vs old chat narratives)

| Old (wrong / superseded) | Current (locked) |
|--------------------------|------------------|
| DRO fragile under PGD | **τ=100 artifact**; fixed **τ=1** makes DRO robust on Adult & Credit α≤0.2 |
| 3 seeds / K=5 / stepped τ | **6 seeds, K=10, τ=1** everywhere |
| IF ≈ 0 / mislabelled | Cosine IF; full **180** IF-attack rows; **mixed** story |
| UTKFace synthetic / blocked | **REAL features**; grid running; **no paper claim** until 90/90 reviewed |
| “6/6 every α” | **α=0.1 is 5/6** (must say) |

---

## F. Still open relative to their asks

| Item | Notes |
|------|--------|
| UTKFace full protocol grid | Local ~**54/90** REAL (DP done, IF running, Combined next) |
| flair2 full GPU showcase | Parked (proven); optional for Aug 10 pixel-level etc. |
| Random-vs-adversarial lead figure | Absolute DP done historically; not on critical path for τ=1 main claim |
| Empirical-radii full table in paper | Q5 appendix path; uniform is SSOT for 540 |
| Constant-predictor plots at meeting quality | Prefer `figD1–D4` regenerated from 540 |

---

## G. Meeting one-pager (what to say if asked)

1. **Manisha’s two tasks:** PGD fairness attacks done; UTKFace real features + experiment running (server staged).  
2. **Kuldeep τ:** fixed to 1 — old “fragile DRO” was temperature.  
3. **Protocol:** n=6, K_inner=10, 540 complete.  
4. **Honest:** Adult DP α=0.1 is 5/6; α≥0.3 below constant predictor on Adult/Credit; LSAC/DP degenerate; IF mixed.  
5. **Verify claims:** we rechecked every number against `canonical_tau1.json`.

**Files:** `docs/MEETING_2026-08-04.md`, `docs/VERIFICATION_REPORT.md`, `docs/KULDEEP_CORRECTION.md`, `figures/fig_tau1_headline.pdf`, `figures/fig_final_wilcoxon_table.pdf`.
