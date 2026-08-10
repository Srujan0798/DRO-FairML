# DRO-FairML — Project STATUS (single source of truth)

_Last updated: 2026-08-10 (SUBMISSION-DAY FINAL). All tasks complete: canonical grid 540+900, all 12 Wave-1 ablations, AL validation A–F, TASK F reproducibility re-run, GPU lane U1–U3, paper+report rebuilt. One human decision pending (see §10). See `docs/SUBMISSION_READY.md` for the final verdict._
**CLEAR:** canonical grid **900/900** (540 n=6 locked + n=10 extension); UTKFace **90/90 REAL**; paper figures/tables; IF metric wins Adult/Credit incl. α=0.3; Adult/DP α=0.1 = **5/6** (marginal — 4/6 under corrected-metric re-run, see §10).

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
- **Adult / DP:** DRO better at every α with p ≤ 0.031 — **not** “6/6 every α”:
  **α=0.1 is 5/6** (seed 2 loses, p=0.0312); **α∈{0.0,0.2,0.3,0.4} are 6/6** (p=0.0156).
- **Adult / Combined:** 6/6 wins at every α, p = 0.016.
- **Credit / DP + Combined:** DRO wins at essentially every cell, p < 0.05
  (Combined α=0.1 is 5/6; others 6/6 on the complete DP+Combined grid).
- **LSAC / Combined:** genuine win, p = 0.016 at α = 0.1 / 0.3 / 0.4 (α=0.2 is 5/6, p=0.031).
- **LSAC / DP:** DEGENERATE NEGATIVE — DRO loses to Naive at every α (0/6 seeds); DRO DP is
  *higher* (worse) than Naive; accuracy is pinned to the majority-class baseline (~0.90) and
  Naive DP is frozen at 0.1827 for α ≥ 0.2. See `docs/LSAC_DEGENERACY.md`.
- **Defensible regime: α ≤ 0.2** on Adult and Credit. At α ≥ 0.3 both methods fall *below* the
  constant-predictor baseline on **Adult (~0.752) and Credit (~0.779)** only → no method claim
  there. **LSAC does not go below** its majority baseline (~0.902); it is **pinned at** it
  (degeneracy), which is a different failure mode — do **not** say “every dataset below.”
- **IF-attack third (COMPLETE — first real numbers):** max |if_clean| ≈ **0.239**.
  Full tables: `results/if_wilcoxon_summary.txt` + `docs/MEETING_2026-08-04.md`.
  **Verdict: MIXED** (not a clean three-attack mirror of DP+Combined).
  - **Adult:** IF metric 6/6 p=0.0156 at α∈{0.1–0.4} (**incl. α=0.3** — Kuldeep-authorized IF claim). α=0 n.s. **DP under IF: win α≤0.2 (6/6); LOSS α=0.3 (1/6, p=0.8906); n.s. α=0.4 (4/6).**
  - **Credit:** IF metric 6/6 p=0.0156 at α≥0.1 (**incl. α=0.3**); DP under IF mostly wins (α=0.1 is **4/6 n.s.**).
  - **LSAC:** IF lose/n.s. α≤0.2; only α∈{0.3,0.4} IF 6/6. DP under IF: **0/6 at α≤0.3**.
  - State **IF metric** wins where supported; do **not** claim “wins on all three attacks on all datasets” for DP.

## 4. Ablations
Historical notes: `docs/reference/ABLATION_STATUS_REPORT.md`. **Canonical claim still
excludes ablations.** Wave-1 re-runs (if/when complete) write **separate** files only:
`results/knn_ablation.json`, `tau_ablation.json`, `lambda_grid.json`,
`random_vs_adversarial.json`, `empirical_radii.json`, `kinner_ablation.json`
via `experiments/run_a*.py` + `run_ablation_parallel.py` (hard-refuses
`canonical_tau1.json` / `utkface_canonical.json`).

## 5. UTKFace (image modality)
**REAL local MPS grid COMPLETE — 90/90 rows (`data_provenance=REAL`).**  
- Features: `data/raw/utkface_features.npz` (23,705 × 512 ResNet18).  
- Output: `results/utkface_canonical.json` + summary `results/utkface_summary.md`.  
- Attacks: dp / if / combined = 30 each (5 α × 6 seeds).  
- **flair2: PROVEN & STAGED, PARKED** — not used for these numbers.  
- **Paper claim:** only after human review of `utkface_summary.md` (do not auto-claim a tabular-style sweep).

## 6. Deliverables status
| Item | State |
|------|-------|
| Canonical full grid (540 rows) | ✅ complete (τ=1, k=10, n=6; IF non-deg max≈0.239) |
| IF-attack third | ✅ complete — **MIXED** |
| Agent V claim fixes | ✅ 5/6, LSAC pinned, figures from 540 |
| Meeting handout (share) | ✅ `docs/MEETING_HANDOUT_2026-08-04.md` |
| Verification LIVE | ✅ `docs/VERIFICATION_REPORT.md` |
| UTKFace REAL 90/90 | ✅ complete; summary `results/utkface_summary.md` (**mixed clean-test**) |
| Paper / report | ✅ Aug 10 narrative + **wired figures/tables** (540-backed); honest UTKFace pilot |
| flair2 | ✅ **UNLOCKED** — 2× L40S, torch 2.6.0+cu124, `docs/FLAIR2_GPU_READY.md` |
| Wave-1 A3 λ/lr grid | ✅ **72/72** `results/lambda_grid.json` + `lambda_grid_summary.md` — λ_init=0.1 improves DP+acc vs default; **no** α=0.3 acc rescue >0.7521 |
| Wave-1 other ablations | 🔄 A4/A5/A1/A2/N5 partial on disk; N2 high-α **LIVE** (orchestrator); separate JSONs only |
| IF@α=0.3 formalization (N4) | ✅ `results/if_wilcoxon_n4_summary.md` (analysis on locked 540) |


## 7. Aug 10 submission checklist
1. ✅ Tabular 540 frozen; do not retrain.
2. ✅ UTKFace 90/90 REAL in repo; paper states **mixed** clean-test (not Adult copy).
3. ✅ Paper/report rebuilt with τ=1 / 5/6 / IF mixed / LSAC degenerate / UTKFace pilot.
4. ✅ Final gate: `make test` (90 pass) && `make validate` (PASS) && `make paper` && `make report`.
5. 🔄 Wave-1 advisor ablations: A3 done; rest in progress (not required to share paper/report).
6. Optional later: flair2 pixel-level experiments if greenlit.


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


## Grok lane update (2026-08-05, HANDOFF_GROK)

- **Phase 0 done:** findings in `docs/HANDOFF_GROK.md` (α=0 disclosure OK; cosine IF flagged for integration; Q6 global cosine attack+eval; training cosine fix; Q7 IF→DP sketch).
- **flair2 U1 DONE:** `results/utkface_flair2.json` (CUDA repro of 90-config protocol; reproducibility summary `results/utkface_reproducibility_summary.md` — max ΔDP < 0.013, all OK).
- **flair2 U2 DONE:** `results/utkface_multigroup.json` (5-race max-min DP eval; 30/30; summary `results/utkface_multigroup_summary.md` — DRO wins multi-group DP at α≥0.2, 6/6 at α=0.4).
- **flair2 U3 DONE:** `results/utkface_pixel_pgd.json` (pixel-space PGD on raw UTKFace JPEGs; 12/12 cells α∈{0.1,0.2}; summary `results/pixel_pgd_summary.md` — DP mixed 4/6 at α=0.1, 2/6 at α=0.2; IF clean 6/6 wins both α; contrast to U1 feature-space in summary).
- **Ops fix (late tick):** `scripts/u12_puller.sh` + `flair2_u12_status.sh` had U3 target **24** (stale) and false-positive process alive flags (checker argv self-match). Corrected to **12** + pgrep bracket detection so puller can finalize/exit. Docs `WAVE1_LIVE_PROGRESS.md` / `UTKFACE_STATUS.md` synced. GPUs idle; no U1/U2/U3 workers.
- **ATTENTION (Grok mistake, 2026-08-05 ~23:01):** During commit hygiene, `git checkout -- results/canonical_tau1.json` restored **HEAD** (252 137 B, **568** rows) over a larger **uncommitted** working-tree file (~517 230 B, **900** rows seen at tick start). Recovery failed (no git object, no /tmp snapshot of the 900-row file). Current disk file = committed baseline only. **GLM / Mac writer:** if that 900-row expansion was progressive fill, re-run/resume from your writer — Grok must not touch this file again.
- Do not rewrite `canonical_tau1.json` / `utkface_canonical.json`.

## 10. Final state (2026-08-10, submission day)

Everything is done except one human decision. **See `docs/SUBMISSION_READY.md`**
for the complete final verdict and `results/task_f_verification_summary.md`
for TASK F details.

- **TASK F (reproducibility re-run): COMPLETE.** `results/canonical_tau1_cosine.json`
  540/540 rows. IF column fixed: noise in 360/540 → real in 540/540. DP/IF
  attacks stable (max drift < 0.008). Combined attack drifts at α≥0.2
  (acc max 7.5%, DP max 8.5%) — the corrected IF metric changes which samples
  the combined attack corrupts.
- **Two n=6 significance flips in the re-run data** (verified with the repo's
  own `compute_wilcoxon`): `adult/dp/α=0.1` 5/6→4/6 (p=0.031→0.109) and
  `lsac/combined/α=0.1` 6/6→4/6 (p=0.016→0.078). Paper/report disclose both as
  marginal. All other cells keep significance.
- **Corrected this week (committed):** 12–40× random-vs-adversarial claim
  (real −3.7× to 1.6×); N1 ρ=0.668→ρ=0.131 (directional, not sig); S flip
  count 6→1; the false "reproduces accuracy exactly / DP shifts O(10⁻⁷)"
  reproducibility sentence replaced with real TASK F deltas.
- **Builds:** `make paper` 393 KiB ✅ · `make report` 320 KiB ✅ · `pytest tests/ -q`
  101 passed ✅ · `make validate` PASS (6/9 DP wins, gate ≥6/9) ✅
- **Pending human decision (Prof. Manisha):** paper reports the locked grid;
  the two α=0.1 cells are flagged marginal. Choose (a) keep locked data with
  caveat disclosed (current) or (b) adopt the re-run file — do not submit
  without this call.
- **Constraints still in force:** do not touch `results/canonical_tau1.json`,
  `results/utkface_canonical.json`; do not merge `canonical_tau1_cosine.json`
  into either. Private repo, no publicity without PI approval.
- **Next:** U4 CelebA optional stretch only; paper/report integration is a separate pass (do not re-edit tex unless Finding 3 cosine sentence missing — already present).
