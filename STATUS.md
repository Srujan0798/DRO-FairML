# DRO-FairML — Project STATUS (single source of truth)

_Last updated: 2026-07-20. Supersedes all prior STATUS / handoff docs._

## 1. What this project is
Implement **DRO-FAIR** (min-max Lagrangian with corruption-calibrated TV uncertainty
sets) and show it is robust to **adversarial** fairness corruption vs the **Naive-FAIR**
baseline. Corruption is a **Fairness-Targeted PGD** attack (not random noise).
Datasets: Adult, Credit, LSAC. Metrics: DP (demographic parity), IF (individual
fairness), accuracy.

## 2. Canonical configuration (locked)
| Param | Value |
|-------|-------|
| tau | 1.0 (fixed) — the old stepped tau=100 was the artifact that made DRO look fragile |
| K_inner | 10 |
| epochs | 60 |
| pgd_steps | 20 |
| seeds | 6 (n≥6 for Wilcoxon p<0.05) |
| lambda_init | 0.0 |
| radii_mode | uniform |
| coordinated | False |

**Grid actually completed: 3 datasets × 5 α × {DP, Combined} × 6 seeds × 2 methods =
360 rows** in `results/canonical_tau1.json` (verify: `python3 -c "import json,collections;
d=json.load(open('results/canonical_tau1.json')); print(len(d), collections.Counter(r['attack']
for r in d))"`). The **IF-attack third (180 rows) was never generated** (the IF metric was
degenerate, ~1e-10) and is cluster-blocked — see §7.

## 3. Verified results (n=6, read from canonical_tau1.json)
- **Adult / DP:** DRO wins at every α (6/6), p ≤ 0.031.
- **Adult / Combined:** 6/6 wins at every α, p = 0.016.
- **Credit / DP + Combined:** DRO wins at essentially every cell, p < 0.05.
- **LSAC / Combined:** genuine win, p = 0.016 at α = 0.1 / 0.3 / 0.4.
- **LSAC / DP:** DEGENERATE NEGATIVE — DRO loses to Naive at every α (0/6 seeds); DRO DP is
  *higher* (worse) than Naive; accuracy is pinned to the majority-class baseline (~0.90) and
  Naive DP is frozen at 0.1827 for α ≥ 0.2. The model collapses to the constant predictor.
  See `docs/LSAC_DEGENERACY.md`.
- **Defensible regime: α ≤ 0.2** on Adult and Credit. At α ≥ 0.3 both methods fall *below* the
  constant-predictor baseline (Adult 0.752, Credit 0.779, LSAC 0.902) → no method claim there.
- **IF:** metric is fixed (cosine-based) in `src/evaluation/metrics.py`, but the 180 IF-attack
  rows are NOT regenerated; IF cells currently read 0.0000 (degenerate pre-fix metric). **No IF
  claim is made** until the cluster re-run. A local proof-of-concept (Adult, α=0.2, 1 config)
  produced `if_clean = 0.0333` — non-degenerate, confirming the fix works on real data
  (evidence: `results/if_poc_adult.json`). The full 180-row sweep still needs the cluster
  (~20 min/config on this CPU → ~60 h; GPU-capable cluster is the only feasible path).

## 4. Ablations
Adjudicated in `docs/ABLATION_STATUS_REPORT.md`: tau / lambda / random-vs-adv dropped with
written reasons; kNN retracted (was actually the Adult IF config, subsumed by the cluster
re-run). None are part of the canonical claim.

## 5. UTKFace (image modality)
**No real UTKFace image experiment has ever been run.** GPU access to flair2.iitgn.ac.in was
never granted (the access request in `docs/EMAIL_TO_SUPIN_GOPI_DRAFT.txt` was drafted but
never sent), so the pipeline substitutes `_make_synthetic_utkface` (random Gaussian 512-d
features) when images are absent. The only UTKFace outputs were **synthetic smoke tests**, not
a real GPU run, and the earlier "DRO inverts on image features" claim is withdrawn. Blocked /
future work. (Full write-up moved to `docs/_archive/UTKFACE_RESULTS_SYNTHETIC_SMOKE_ONLY.md`.)

## 6. Deliverables status
| Item | State |
|------|-------|
| Canonical (DP+Combined, 360 rows) | ✅ committed |
| IF-attack third (180 rows) | ❌ cluster-blocked (G1) |
| Tables / figures / both PDFs | ✅ regenerated from canonical (`make paper` / `make report`) |
| Kuldeep correction note | ✅ `docs/KULDEEP_CORRECTION.md` (draft — human reviews & sends) |
| UTKFace | ⛔ no real run; synthetic-only; blocked |

## 7. What remains (only two items)
1. **IF cluster re-run (G1):** generate the 180 IF-attack rows on a cluster
   (`scripts/run_if_rerun_cluster.sh`, resume-safe — appends only missing keys), then
   regenerate tables/figures/PDFs. ~15 h single-CPU; GPU not required. The IF metric code fix
   is verified (`tests/test_metrics.py` pass).
2. **UTKFace decision (human):** either send the flair2 access email or formally drop UTKFace
   from scope.

## 8. How to run / verify
```bash
make paper && make report     # rebuild both PDFs from canonical
make validate                 # Wilcoxon: DP wins 6/9 at p<0.05; LSAC not significant; IF=0.0000
make results && make deliverables   # regenerate tables + figures
```

## 9. Constraints
- Private repo, professor only. No publicity.
- No oracle leak: DRO knows only α (+ known attack structure for empirical radii) — never the
  true per-sample corruption mask.
