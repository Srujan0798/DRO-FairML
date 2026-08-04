# DRO-FairML — Project STATUS (single source of truth)

_Last updated: 2026-08-04 (Agent I meeting brief). Supersedes all prior STATUS / handoff docs.
**CLEAR:** canonical grid **540/540** (IF 180); Agent H finalize complete; first real IF numbers below._

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

**Full grid complete: 3 datasets × 5 α × {DP, IF, Combined} × 6 seeds × 2 methods =
540 rows** in `results/canonical_tau1.json` (unique keys 540). IF metric is cosine-based
and **non-degenerate** under attack=if (max |if_clean| ≈ **0.239**). Verify anytime:
`python3 -c "import json,collections; d=json.load(open('results/canonical_tau1.json')); print(len(d), dict(collections.Counter(r['attack'] for r in d)))"`.

## 3. Verified results (n=6, read from canonical_tau1.json)
- **Adult / DP:** DRO wins every α at p ≤ 0.031 (**α=0.1 is 5/6**, seed 2 loses; **others 6/6**).
- **Adult / Combined:** 6/6 wins at every α, p = 0.016.
- **Credit / DP + Combined:** DRO wins at essentially every cell, p < 0.05
  (Combined α=0.1 is 5/6; others 6/6 on the complete DP+Combined grid).
- **LSAC / Combined:** genuine win, p = 0.016 at α = 0.1 / 0.3 / 0.4 (α=0.2 is 5/6, p=0.031).
- **LSAC / DP:** DEGENERATE NEGATIVE — DRO loses to Naive at every α (0/6 seeds); DRO DP is
  *higher* (worse) than Naive; accuracy is pinned to the majority-class baseline (~0.90) and
  Naive DP is frozen at 0.1827 for α ≥ 0.2. See `docs/LSAC_DEGENERACY.md`.
- **Defensible regime: α ≤ 0.2** on Adult and Credit. At α ≥ 0.3 both methods fall *below* the
  constant-predictor baseline on **Adult (0.752) and Credit (0.779)** → no method claim there.
- **IF-attack third (COMPLETE — first real numbers):** max |if_clean| ≈ **0.239**.
  Full tables: `results/if_wilcoxon_summary.txt` + `docs/MEETING_2026-08-04.md`.
  **Verdict: MIXED** (not a clean three-attack mirror of DP+Combined).
  - **Adult:** IF metric 6/6 p=0.0156 at α∈{0.1–0.4}; α=0 n.s. **DP under IF: win α≤0.2 (6/6);
    LOSS α=0.3 (1/6, p=0.8906); n.s. α=0.4 (4/6).**
  - **Credit:** IF metric 6/6 p=0.0156 at α≥0.1; DP under IF mostly wins (α=0.1 is **4/6 n.s.**).
  - **LSAC:** IF lose/n.s. α≤0.2; only α∈{0.3,0.4} IF 6/6. DP under IF: **0/6 at α≤0.3**.
  - Do **not** claim “wins on all three attacks on all datasets.”

## 4. Ablations
Adjudicated in `docs/ABLATION_STATUS_REPORT.md`: tau / lambda / random-vs-adv dropped with
written reasons; kNN retracted. None are part of the canonical claim.

## 5. UTKFace (image modality)
**Real features on disk; local MPS REAL grid in progress — not a paper claim yet.**  
`data/raw/utkface_features.npz` (provenance REAL). Runner writes
`results/utkface_canonical.json` (all rows `data_provenance=REAL`). Progress is
**partial** (attack=`dp` only so far; see `docs/UTKFACE_STATUS.md` for live count).  
flair2 SSH remains optional for a faster full grid. **No paper claim** until a
reviewed multi-seed multi-attack subset exists.

## 6. Deliverables status
| Item | State |
|------|-------|
| Canonical full grid (540 rows) | ✅ complete (unique 540; τ=1 / k_inner=10 / epochs=60 uniform; max\|if\|≈0.239) |
| IF-attack third (180 rows) | ✅ complete — **first real IF numbers; MIXED** vs DP story (see §3) |
| Agent H finalize | ✅ `results/if_wilcoxon_summary.txt`; tables; `paper/main.pdf` + `report/report.pdf` |
| Meeting brief (4pm) | ✅ FINAL `docs/MEETING_2026-08-04.md` (honest 5/6 + IF MIXED) |
| Kuldeep correction note | ✅ `docs/KULDEEP_CORRECTION.md` (draft — human reviews & sends) |
| UTKFace full grid | 🔄 **local MPS REAL partial** (`utkface_canonical.json`, dp-only so far); **no paper claim** |
| Note | H log: `compute_canonical_wilcoxon.py` once hit `ModuleNotFoundError: experiments`; IF summary + auto tables OK. L: `PYTHONPATH=.` if full wilcoxon artifact needed. |

## 7. What remains (Aug 10)
1. **J more:** continue repo professionalize (archive dead paths, polish entrypoints).
2. **K:** paper/report prose to 540-row tables + honest IF narrative (**MIXED**: Adult/Credit α≤0.2 yes; Adult α≥0.3 DP-under-IF loss; LSAC IF no; Adult/DP α=0.1 = 5/6).
3. **L re-audit:** ship gate on full 540 + H artifacts (`docs/VERIFICATION_REPORT.md`).
4. **M:** UTKFace — finish local REAL multi-attack grid or formal Aug 10 scope-out (row count only until then).
5. Optional: fix PYTHONPATH for `experiments/compute_canonical_wilcoxon.py` in finalize script.

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
