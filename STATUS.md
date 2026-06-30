# DRO-FairML — Project STATUS (single source of truth)

_Last updated: 2026-06-27. Supersedes the archived handoff docs in `docs/archive/`._

## 1. What this project is
Implement **DRO-FAIR** (Algorithm 1, min-max Lagrangian with corruption-calibrated TV
uncertainty sets) and show it is robust to **adversarial** fairness corruption, versus the
**Naive-FAIR** baseline. Corruption is a **Fairness-Targeted PGD** attack (not random noise).
Datasets: Adult, Credit, LSAC. Metrics: DP (demographic parity), IF (individual fairness), accuracy.

Professor: Manisha Padala. Technical reviewer: Kuldeep. Trigger directive (Jun 2): *"Check the
adversarial attack on DP and improve it. Then redo all experiments."*

## 2. Canonical configuration (locked — do not change)
| Param | Value | Why |
|-------|-------|-----|
| tau | **1.0 fixed (all α)** | old stepped tau=100 was the artifact that made DRO look fragile (Q12) |
| K_inner | **10** | paper-mandatory |
| epochs | **60** | paper-mandatory |
| pgd_steps | 20 | full attack strength |
| seeds | **6** | n≥6 needed for Wilcoxon p<0.05 (Q9) |
| lambda_init | 0.0 | paper spec; exposed only for Q1 ablation; NOT used in inner gradient |
| lambda_max | 1.5 | all datasets |
| radii_mode | uniform (main) / empirical (Q5 companion) | see §4 |
| coordinated | **False** (main canonical) / **True** (empirical companion) | see §4 |

Grid = 3 datasets × 5 α × 3 attacks × 2 methods × 6 seeds = **540 rows** → `results/canonical_tau1.json`.

## 3. Verified-correct (audit 2026-06-27, line-by-line)
- **Integration**: train on poisoned data, evaluate fairness on **clean test** — `run_fairness_pgd.py:80,102`.
- **DP attack improved** (madam Jun 2): direct gradient ascent on |p0−p1|, NOT BCE — `adversarial.py:589`; exact analytical DP label-gradient — `adversarial.py:241`.
- **IF attack** within-group k-NN (Q6), k∈{5,10,15} ablation — `adversarial.py:297`.
- **DRO trainer**: tau-multiply `σ(τ·f)` (`dro_fair.py:289`); step order θ→λ→p (`:298-341`); λ NOT in inner gradient (`:327`); radii ρ_DP=α/((1−α)π+α), ρ_IF=2α−α² (`:103-107`); α=0 guard fixing the LSAC anomaly Q4 (`:321`).
- **Tests**: 60/60 pass.

## 4. Q5 empirical radii — FIXED 2026-06-27 (commit 44c6a31)
Earlier the empirical companion was a **no-op duplicate**: clean `a_val` always overrode
`radii_mode` in `_compute_radii`, so empirical produced identical radii to uniform. Fixes:
1. `_compute_radii`: empirical now takes precedence over `a_val` → the known-attack inversion
   `pi_clean = pi_obs + 0.4α` actually computes the radii. Uniform/canonical path unchanged.
2. `run_canonical_empirical.py`: `coordinated=True` (was False) — REQUIRED so the 70/30
   minority-targeting matches the inversion assumption; otherwise it inverts the wrong attack.

**Q5 result**: the empirical inversion recovers radii nearly identical to clean validation —
i.e. *you don't need clean validation if you know the attack*. Standalone study, distinct from
the coordinated=False main canonical.

## 5. Headline result (real 6-seed canonical, Adult DP attack)
DRO ≥ Naive on **both** DP and accuracy at **every** α. Advantage on DP grows with α.

| α | naive DP | DRO DP | DRO acc | regime |
|---|----------|--------|---------|--------|
| 0.0 | 0.149 | 0.143 | 0.815 | ✅ acc≥0.78 |
| 0.1 | 0.203 | 0.200 | 0.818 | ✅ acc≥0.78 |
| 0.2 | 0.245 | 0.233 | 0.759 | ⚠ beats constant predictor (0.752), <0.78 |
| 0.3 | 0.285 | 0.261 | 0.676 | ❌ below constant predictor |
| 0.4 | 0.314 | 0.286 | 0.561 | ❌ below constant predictor |

**Defensible regime = α≤0.2** (DRO beats the constant-label predictor). At α≥0.3 the constant
predictor wins due to a 30–40% label-corruption ceiling — not fixable by tau or lambda
(verified by full tau {1,5,10,20,100} and lambda {init×lr} ablations). Figures/report must state
the acc≥0.78 bound holds strictly only to **α≤0.1**.

## 6. Deliverables status
| Item | State |
|------|-------|
| Canonical 540 (uniform) | running — see `results/canonical_tau1.json` |
| Empirical 270 (Q5, fixed config) | running — `results/canonical_tau1_empirical.json` |
| Lambda grid Q1 (72) | ✅ done |
| k-NN ablation Q6 (k=5/10/15, all datasets) | ✅ done (120 rows) |
| tau ablation Q12 (1/5/10/20/100) | tau1/10/20/100 done; tau5 finishing |
| n=6 Wilcoxon (all datasets) | auto-regen by orchestrator at 540 |
| Final figures (from final canonical) | auto-regen by orchestrator at 540 |
| Report + paper PDFs | rebuild clean (tectonic); auto-regen at 540 |
| UTKFace (Q13) | **BLOCKED** — flair2.iitgn.ac.in unresponsive; Prof. aware (Jun 19); local smoke confirms pipeline |

## 7. How to run / recover
```bash
# Canonical (resume-safe)
nohup python3 -u experiments/run_canonical.py > logs/canonical_resume.log 2>&1 &
# Empirical companion (resume-safe, fixed config)
nohup python3 -u experiments/run_canonical_empirical.py > logs/empirical_resume.log 2>&1 &
# Orchestrator: polls canonical→540, then auto regen figures+wilcoxon+report+PDFs
nohup bash scripts/final_delivery_orchestrator.sh > /dev/null 2>&1 &
# Monitor
python3 -c "import json;print(len(json.load(open('results/canonical_tau1.json'))),'/540')"
```

## 8. Constraints
- Private repo, professor only. No publicity.
- No oracle leak: DRO knows only α (+ known attack structure for empirical radii) — never the
  true per-sample corruption mask.
