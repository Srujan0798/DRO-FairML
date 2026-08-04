# DRO-FairML — Project STATUS (single source of truth)

_Last updated: 2026-08-04. Supersedes all prior STATUS / handoff docs.
Light refresh: §6–§7 note local IF sweep in progress toward 540; DP+Combined still complete._

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

**DP + Combined completed: 3 datasets × 5 α × {DP, Combined} × 6 seeds × 2 methods =
360 rows** in `results/canonical_tau1.json`. The **IF-attack third** (target +180 → **540**
total) was blocked for a long time by a degenerate IF metric (~1e-10). The metric is fixed
(cosine-based); a **local parallel IF sweep is in progress** on 2026-08-04
(`experiments/run_if_parallel.py`) — partial IF rows are non-degenerate when present.
Do **not** treat IF as complete until `len(canonical_tau1.json) == 540` and IF count is 180.
Verify anytime:
`python3 -c "import json,collections; d=json.load(open('results/canonical_tau1.json')); print(len(d), dict(collections.Counter(r['attack'] for r in d)))"`.

## 3. Verified results (n=6, read from canonical_tau1.json)
- **Adult / DP:** DRO wins every α at p ≤ 0.031 (**α=0.1 is 5/6**, seed 2 loses; **others 6/6**).
- **Adult / Combined:** 6/6 wins at every α, p = 0.016.
- **Credit / DP + Combined:** DRO wins at essentially every cell, p < 0.05
  (Combined α=0.1 is 5/6; others 6/6 on the complete DP+Combined grid).
- **LSAC / Combined:** genuine win, p = 0.016 at α = 0.1 / 0.3 / 0.4 (α=0.2 is 5/6, p=0.031).
- **LSAC / DP:** DEGENERATE NEGATIVE — DRO loses to Naive at every α (0/6 seeds); DRO DP is
  *higher* (worse) than Naive; accuracy is pinned to the majority-class baseline (~0.90) and
  Naive DP is frozen at 0.1827 for α ≥ 0.2. The model collapses to the constant predictor.
  See `docs/LSAC_DEGENERACY.md`.
- **Defensible regime: α ≤ 0.2** on Adult and Credit. At α ≥ 0.3 both methods fall *below* the
  constant-predictor baseline on **Adult (0.752) and Credit (0.779)** → no method claim there.
  **LSAC is not below baseline:** under DP attack accuracy stays **pinned at** the majority rate
  (~0.902 vs 0.9016), the same degeneracy as above — not an accuracy collapse below constant.
- **IF:** metric is fixed (cosine-based) in `src/evaluation/metrics.py`. PoC:
  `results/if_poc_adult.json` (`if_clean ≈ 0.0333`). As of **2026-08-04**, a **local IF
  sweep is in progress** (`experiments/run_if_parallel.py` → `canonical_tau1.json`). Live
  snapshot **2026-08-04 ~13:46 IST**: **82/180 IF rows** (total **442/540**); Adult IF complete
  (60/60), Credit in progress (~22), LSAC not started. IF-attack `if_clean` is
  **non-degenerate** (max |if_clean| ≈ **0.098**, all IF rows ≫ 1e-6). DP/Combined rows still
  show IF *metric column* ~0 because those attacks do not stress IF. **No full-grid IF claim
  until IF = 180 / total 540 and tables/figures are regenerated.** Do not say “IF never
  generated” or “IF cells are 0.0000” for the IF-attack third — that language is stale.
  Live: `docs/LOOP_STATUS.md`.

## 4. Ablations
Adjudicated in `docs/ABLATION_STATUS_REPORT.md`: tau / lambda / random-vs-adv dropped with
written reasons; kNN retracted (was actually the Adult IF config, subsumed by the cluster
re-run). None are part of the canonical claim.

## 5. UTKFace (image modality)
**Real features now on disk; full multi-seed attack grid not yet run.** On 2026-08-04 local Mac
(MPS): images downloaded, `data/raw/utkface_features.npz` extracted (X=23705×512), and a
**timing probe** wrote `results/utkface_timing_probe.json` with `data_provenance=REAL`
(attack=dp, α=0, seed=0 only; ~24s wall). That is **not** a full experiment — do not claim
UTKFace results in the paper. Historical synthetic-only smoke tests remain archived
(`docs/_archive/UTKFACE_RESULTS_SYNTHETIC_SMOKE_ONLY.md`). Full grid deferred until IF sweep
releases CPU cores (load was ~33 during 10-worker IF). See `docs/UTKFACE_STATUS.md`.

## 6. Deliverables status
| Item | State |
|------|-------|
| Canonical (DP+Combined, 360 rows) | ✅ committed |
| IF-attack third (180 rows) | 🔄 **local sweep in progress** (2026-08-04; path **→540** total rows). Not complete until IF=180. |
| Tables / figures / both PDFs | ✅ for DP+Combined; **re-run after IF hits 540** before claiming full-grid IF |
| Kuldeep correction note | ✅ `docs/KULDEEP_CORRECTION.md` (draft — human reviews & sends) |
| UTKFace | 🔄 **real features + 1-config MPS probe** (`docs/UTKFACE_STATUS.md`); full grid still pending |

## 7. What remains (only two items)
1. **IF local sweep → 540 (G1):** finish the 180 IF-attack rows
   (`experiments/run_if_parallel.py` or `scripts/run_if_rerun_cluster.sh` — both resume-safe),
   then **Agent H** finalize (`./scripts/agent_h_finalize.sh`) and regenerate tables/figures/PDFs.
   Metric fix verified (`tests/test_metrics.py`). Live progress: `docs/LOOP_STATUS.md`.
2. **UTKFace full grid (or formal drop):** real features already local; after IF frees cores,
   run multi-α/seed protocol on MPS — or drop from Aug 10 scope (`docs/UTKFACE_STATUS.md`).

## 8. How to run / verify
```bash
bash data/download_data.sh --verify   # or: make data
make test
make paper && make report     # rebuild both PDFs from current canonical
make validate                 # Wilcoxon / consistency on current results
make results && make deliverables   # regenerate tables + figures
# Full repro path: see README.md "How to reproduce"
```


## 9. Constraints
- Private repo, professor only. No publicity.
- No oracle leak: DRO knows only α (+ known attack structure for empirical radii) — never the
  true per-sample corruption mask.
