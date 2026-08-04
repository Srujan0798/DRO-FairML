# MASTER PROTOCOL — Aug 10 Submission (written 2026-08-04, 7 days out)

**This document supersedes all prior plans.** It is the single execution protocol until
submission. Basis: full audit of every file + line-by-line comparison against every
recorded request from Prof. Manisha Padala and Kuldeep (chat log, May 19 – Jun 30).

**Honest completion estimate: core science ~70%, advisor-ask coverage ~50%, paper
assembly ~40%.** The 540-row tabular grid and 90-row REAL UTKFace grid are solid and
committed. But multiple explicit advisor requests were "dropped/deferred" instead of
delivered, the paper PDF contains no figures and no numeric tables, and the compute
that made those drops necessary (no GPU, slow laptop) **no longer exists as an excuse**
— we now have a 14-core M4 Pro (21 s/config measured) and a working flair2 node with
2× NVIDIA L40S.

---

## PART 1 — BRUTAL GAP AUDIT (every recorded ask → delivered or not)

| # | Who / when | Exact ask | Current state | Verdict |
|---|---|---|---|---|
| 1 | Manisha, May 19 | "implement pgd for fairness metrics (Both DP and IF, only DP, only IF)" | FairnessTargetedPGD dp/if/combined, tested | ✅ DONE |
| 2 | Manisha, May 19 | "Set up an experiment for the **UTKFace dataset in the server**" | Ran locally on Mac (REAL, 90/90). **Never ran on flair2.** Server now works; torch not installed | ❌ **GAP — literal ask was the server** |
| 3 | Manisha, Jun 2 | "Check the adversarial attack on DP and improve it. Then, redo all the experiments" | Attack improved (direct DP gradient); 540-row canonical redone | ✅ DONE |
| 4 | Kuldeep, Jun 9 | "Q1: try different initial value of lambdas, learning rates… hyperparameter tuning" | Lambda grid **26/720, crashed, DROPPED**. Data deleted from tree | ❌ **GAP — explicit ask, not delivered** |
| 5 | Kuldeep, Jun 9 | "For if attack we have to do **ablation study for different k 5,10,15**" | k=10 done (canonical); k=5/k=15 backfill "IN PROGRESS" then data deleted. **No k-ablation exists** | ❌ **GAP — explicit ask, not delivered** |
| 6 | Kuldeep, Jun 9 | "we fix tau for all alpha. Here we can use **different tau for ablation study**" | Tau ablation DROPPED (old files confounded τ with k_inner, no LSAC). Data deleted | ❌ **GAP — explicit ask, not delivered** |
| 7 | Kuldeep, Jun 9 (Q5) | "if attack is known then we can use this approximation" (empirical radii) | Implemented in code (`radii_mode='empirical'`); experiment 29/270, DEFERRED, data deleted | ❌ **GAP — code exists, experiment doesn't** |
| 8 | Kuldeep, Jun 16/30 | Random vs adversarial comparison ("12–40×" was quoted to him) | Old Jun-9 data predates canonical protocol; DEFERRED; deleted. **The 12–40× claim has no canonical-protocol backing** | ❌ **GAP — quoted number unsupported** |
| 9 | Kuldeep, Jun 30 | "check accuracy dp and if of constant predictor" | Done, per-dataset, in docs + figures | ✅ DONE |
| 10 | Kuldeep, Jun 30 | IF plots per attack type, honest | Real IF metric fixed; 180 IF rows; honest MIXED story documented | ✅ DONE |
| 11 | Kuldeep, Jun 9 (Q4) | LSAC α=0 anomaly — "expected or another bug?" | Diagnosed (LSAC/DP degenerate; radii-on-imbalance). **No principled fix tested.** α=0 DRO≠Naive objective question has a stated position but no experiment | ⚠️ **PARTIAL — diagnosis only** |
| 12 | Kuldeep, Jun 9 (Q9) | "6 seeds now, or is 3 acceptable…?" (n for publishable) | n=6 everywhere. Min attainable p = 0.0156. n=10 was never evaluated; hardware now makes it ~2 h | ⚠️ **UPGRADE AVAILABLE** |
| 13 | Kuldeep, Jun 30 | "verify all the claims" | Verification pipeline exists; claims traced | ✅ DONE (maintain) |
| 14 | — (implied by "submission under professor") | A paper PDF that actually shows results | **paper/main.tex has 0 `\includegraphics`, 0 wired tables; auto_generated/*.tex orphaned; both appendix files (q1_lambda, q5_empirical) NOT `\input`.** The paper is prose-only | ❌ **CRITICAL GAP** |

**Score: 5 delivered, 6 hard gaps, 2 partial, 1 upgrade.** The 6 hard gaps are all now
cheap: every missing ablation is ≤ 2 h wall-clock on the Mac at measured 21 s/config.
There is no remaining excuse.

---

## PART 2 — COMPUTE INVENTORY (what makes this all feasible)

| Resource | State | Throughput |
|---|---|---|
| Mac M4 Pro, 14 cores, 24 GB, MPS | Proven | **21 s/tabular config**; 10-worker parallel driver pattern proven (`experiments/run_if_parallel.py` — 157 configs in ~40 min) |
| flair2.iitgn.ac.in | SSH key auth working (`ssh flair2`); 2× L40S 46 GB, driver 570 fixed; code + real UTKFace features already at `/data/srujan.sai/DRO-FairML-run/`; **torch NOT installed** (node behind SSL firewall — offline wheel install required; 830 MB wheelhouse already on Mac, missing the 14 nvidia-cu12 wheels ≈ 1.7 GB) | 2× 46 GB GPU — pixel-level PGD and big batches become possible |

**flair2 unlock (Agent G, Day 0–1, run on ethernet or overnight):**
```bash
# on Mac — finish the nvidia wheels (exact pins are in git: docs history / torch 2.6.0 metadata)
python3 -m pip download --platform manylinux2014_x86_64 --python-version 310 \
  --implementation cp --abi cp310 --only-binary=:all: --no-deps \
  nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cuda-cupti-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 \
  nvidia-cublas-cu12==12.4.5.8 nvidia-cufft-cu12==11.2.1.3 \
  nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 \
  nvidia-cusparse-cu12==12.3.1.170 nvidia-cusparselt-cu12==0.6.2 \
  nvidia-nccl-cu12==2.21.5 nvidia-nvtx-cu12==12.4.127 \
  nvidia-nvjitlink-cu12==12.4.127 triton==3.2.0 -d wheelhouse
rsync -az wheelhouse/ flair2:/data/srujan.sai/wheelhouse/
ssh flair2 'cd /data/srujan.sai/DRO-FairML-run && python3 -m venv venv_gpu && \
  ./venv_gpu/bin/pip install --no-index --find-links /data/srujan.sai/wheelhouse \
  torch torchvision numpy scipy scikit-learn pandas && \
  ./venv_gpu/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"'
# gate: must print "True 2" before anything runs there
```

**Parallel-driver pattern (reuse for every ablation below):** copy
`experiments/run_if_parallel.py`, change the config enumeration, keep: isolated worker
processes calling `run_single_experiment` (no file I/O in workers), parent-only atomic
merge with checkpoint after each result, resume-safe key set, full provenance on every
row. **Never run two writers against one JSON file.**

---

## PART 3 — THE 7-DAY EXECUTION PLAN

Waves, not days — each wave gates the next. Agents in one wave run in parallel.
**Global rules for every agent:** (1) provenance fields on every row (tau, k_inner,
epochs, pgd_steps, seed, n_seeds_planned, radii_mode, lambda_init, coordinated);
(2) `python3 -m pytest tests/ -q` green before every commit; (3) never touch a JSON a
live process is writing (`ps aux | grep run_`); (4) one writer per results file;
(5) report negative results as negative — the τ=100 lesson: honest > pretty;
(6) read `docs/reference/ARCHIVE_POLICY.md` before deleting anything.

### WAVE 1 (Day 0–1) — close every advisor gap. All on Mac, all parallel.

**AGENT P — Paper surgery (CRITICAL PATH — the deliverable itself)**
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_PROTOCOL_AUG10.md Part 1
row 14, docs/reference/PAPER_FINALIZATION_CHECKLIST.md (git history if deleted), and
paper/main.tex.

paper/main.pdf currently contains ZERO figures and ZERO numeric tables. Fix:
1. Wire the main results table: \input{../paper/auto_generated/tabular_results} (or move
   into sections/results.tex) — the reader must see the Adult/Credit/LSAC grid.
2. Wire paper/auto_generated/wilcoxon.tex and key_findings.tex where they belong.
3. Add figures to results.tex: fig1_main_results, fig2_dp_reduction_heatmap,
   fig4_significance_matrix, fig5_accuracy_fairness_tradeoff, fig7_summary_win_rates
   (\includegraphics{../figures/...}, with captions stating n=6, τ=1, defensible regime).
4. \input the two orphaned appendices (appendix_q1_lambda, appendix_q5_empirical) after
   \appendix — they will be refreshed by Agents A3/A5's new data in Wave 2.
5. Add a UTKFace subsection to results.tex from results/utkface_summary.md numbers ONLY:
   DP mostly non-significant except α=0.4 (6/6, p=0.016, DP & Combined); IF violation
   consistently lower for DRO; accuracy never worse. Frame as different-modality finding.
6. α=0 treatment (Kuldeep Q4): add explicit statement that DRO and Naive optimize
   different objectives at α=0 (tilted risk vs BCE), so α=0 cells are excluded from
   attack-robustness win counts; cite the numbers either way.
7. Limitations: LSAC/DP degeneracy (cite docs/LSAC_DEGENERACY.md), α≥0.3 constant-
   predictor bound (Adult+Credit; LSAC pinned AT baseline, not below), IF story MIXED.
8. make paper && make report must build; visually check every figure/table renders.
Commit per logical step. Every number traceable to results/*.json.
```

**AGENT A1 — kNN ablation k∈{5,15} (Kuldeep's explicit ask #5)**
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read Part 2 "parallel-driver pattern".
Canonical k=10 IF rows already exist inside results/canonical_tau1.json — do NOT re-run them.
Create experiments/run_knn_ablation_parallel.py from run_if_parallel.py: attack='if',
k_inner ∈ {5,15}, 3 datasets × 5 alphas × 6 seeds × 2 methods = 360 configs → NEW file
results/knn_ablation.json (never write canonical_tau1.json). ~40–80 min at 10 workers.
Then produce results/knn_ablation_summary.md: per (dataset, α): DP/IF/acc at k=5/10/15,
paired Wilcoxon k=5 vs k=10 vs k=15. The deliverable sentence Kuldeep wants: does the IF
attack's strength depend on k? Write the honest answer either way. Refresh
paper/sections/ appendix or results text via Agent P. Commit data + summary.
```

**AGENT A2 — Tau ablation τ∈{10,100} (Kuldeep's explicit ask #6, clean this time)**
```
Same pattern. τ=1 rows = canonical (do not re-run). New runs: τ ∈ {10,100}, attack='dp',
3 datasets × 5 alphas × 6 seeds × 2 methods = 360 configs → results/tau_ablation.json.
CRITICAL: k_inner=10 everywhere (the old ablation was dropped precisely because it
confounded τ with k_inner — do not repeat that), full provenance, include LSAC.
Deliverable: results/tau_ablation_summary.md — the τ=100 artifact demonstrated CLEANLY
on all three datasets: DP(τ=100) vs DP(τ=1) per α, with the DRO-loses-at-τ=100 /
DRO-wins-at-τ=1 flip shown as the paper's motivating ablation. Feeds Agent P's paper
section on the central finding. Commit data + summary.
```

**AGENT A3 — Lambda/LR scoped grid (Kuldeep's explicit ask #4)**
```
Same pattern. Scope (pathology-aware — the old 720-grid died on a λ0=1.0 config that ran
17.9 h; EXCLUDE λ0=1.0): λ_init ∈ {0.0, 0.01, 0.1} × lr_λ ∈ {0.001, 0.005} on Adult,
attack='dp', α ∈ {0.2, 0.3}, 6 seeds, DRO only = 72 configs → results/lambda_grid.json.
~30 min. Deliverable: results/lambda_grid_summary.md + refreshed
paper/sections/appendix_q1_lambda.tex — does any (λ_init, lr_λ) beat the default
(0.0, 0.005) on DP without accuracy loss, and does anything rescue α=0.3 above the
constant predictor (0.7521)? Expected honest answer per old partial data: no — say so
with full-grid evidence. Commit.
```

**AGENT A4 — Random vs adversarial under canonical protocol (backs the quoted 12–40×)**
```
Same pattern. RandomCorruptor vs FairnessTargetedPGD(dp), canonical config, 3 datasets ×
α ∈ {0.1, 0.2} × 6 seeds × 2 methods × 2 corruptions = 144 configs →
results/random_vs_adversarial.json. ~50 min. Deliverable: the multiplier table —
ΔDP(adversarial)/ΔDP(random) per dataset/α with CIs. The "12–40×" number was quoted to
Kuldeep on Jun 16 from pre-canonical data; this either substantiates it or corrects it —
if the real multiplier differs, the paper states the REAL one. Commit.
```

**AGENT A5 — Empirical radii Q5 (Kuldeep's ask #7)**
```
Same pattern. Adult only (scoped): radii_mode='empirical' + coordinated=True vs the
canonical uniform rows, attack='dp', 5 alphas × 6 seeds × 2 methods = 60 new configs →
results/empirical_radii.json. ~25 min. Deliverable: uniform-vs-empirical comparison
table + refreshed paper/sections/appendix_q5_empirical.tex. Honest question: does
attack-aware radii calibration improve DRO under coordinated corruption? Commit.
```

**AGENT G — flair2 unlock (runs alongside; ethernet or overnight)**
Commands in Part 2. Gate: `torch.cuda.is_available() → True, device_count → 2`.
Do not start Wave 2's server work until this gate passes.

### WAVE 2 (Day 2–3) — the upgrades the new hardware makes possible

**AGENT S — Seeds n=6 → n=10 (tabular + UTKFace)**
```
Extend the canonical grids with seeds 6–9. Tabular: 3 datasets × 3 attacks × 5 α ×
4 new seeds × 2 methods = 360 configs → APPEND to results/canonical_tau1.json via the
resume-safe pattern (missing-key enumeration; single writer; verify 900 rows after).
UTKFace: same +4 seeds = 60 rows → utkface_canonical.json (150 total). ~2–3 h Mac.
Then REGENERATE every Wilcoxon/table/figure at n=10 (min attainable p drops 0.0156 →
0.001) and have Agent P update every "n=6" caption/claim to n=10. This directly answers
Kuldeep's Q9 ("or push for more?"). If any cell flips significance at n=10, the paper
reports the n=10 truth. Commit data first, artifacts second.
```

**AGENT U — UTKFace ON THE SERVER (Manisha's literal May-19 ask #2) + pixel-PGD stretch**
```
Requires Agent G's gate. (a) Run the identical UTKFace grid on flair2
(deploy script exists: scripts/deploy_utkface_flair2.sh; runner supports --device cuda)
→ results/utkface_flair2.json; verify per-cell agreement with the Mac MPS run (same
seeds ⇒ near-identical numbers = a reproducibility appendix row, and the literal
"in the server" ask is finally TRUE). (b) STRETCH — the experiment only a 46 GB GPU can
do: pixel-space PGD on raw UTKFace images through ResNet18 (image-space fairness attack,
vs our feature-space attack). src/corruption/image_pgd.py exists in git history
(restore per ARCHIVE_POLICY: git log --all -- src/corruption/image_pgd.py). Scope: Adult
protocol transplanted — α ∈ {0.1, 0.2}, 6 seeds, dp attack, 2 methods. If it lands, the
paper gains a "feature-space vs pixel-space attack" section no reviewer will have seen
from a course project. If it can't land by Day 4, cut it cleanly — (a) alone closes the ask.
```

**AGENT L2 — LSAC degeneracy: test the principled fix (Kuldeep Q4, completed honestly)**
```
docs/LSAC_DEGENERACY.md diagnoses radii blow-up on the ~90/10 imbalanced minority group.
Test the fix the diagnosis implies — empirical/clamped per-group radii
(rho_dp[j] capped, or radii_mode='empirical') on LSAC, attack='dp', 5 α × 6 seeds ×
2 methods = 60 configs → results/lsac_radii_fix.json. THE RULE: this is hypothesis
testing, not tuning-until-win. If the fix un-degenerates LSAC (accuracy off the 0.9016
pin, DP unfrozen), the paper upgrades LSAC from "degenerate, excluded" to "fixed by
attack-aware radii" — a genuine contribution. If it doesn't, the limitation stands with
evidence. Either outcome is publishable; only the untested state is not. Commit.
```

### WAVE 3 (Day 4–5) — integration and gate

**AGENT I2 — Full integration**: every Wave-1/2 result wired into paper + report;
regenerate all figures/tables from the final data (n=10 if Agent S landed); both PDFs
build; report and paper tell the SAME story with the same numbers.

**AGENT V2 — Final verification gate (the Kuldeep protocol, ask #13)**
```
Adversarial pass over EVERYTHING: recompute every mean/win-count/p-value in paper,
report, STATUS.md, and all summaries directly from the results/*.json files. Trace the
12–40× (or corrected) multiplier, every ablation sentence, every caption n. Produce
docs/VERIFICATION_FINAL.md with claim → source-file → recomputed → MATCH/MISMATCH.
ZERO unresolved mismatches ship. Run make test / make validate / make paper /
make report as the mechanical gate.
```

### Day 6 — buffer, advisor pre-read (send Manisha + Kuldeep the PDFs + honest
one-pager), fix whatever they flag. **Day 7 — submit.**

---

## PART 4 — DEFINITION OF 100%

- [ ] Every row of the Part-1 gap table flipped to ✅ (or a written advisor-visible
      reason where a stretch was cut)
- [ ] Paper PDF shows: main table, ≥5 figures, Wilcoxon table, tau ablation, kNN
      ablation, lambda appendix, Q5 appendix, UTKFace section, α=0 statement, honest
      limitations
- [ ] n=10 seeds (or a documented decision to stay at 6), all stats regenerated
- [ ] UTKFace has run on flair2 (the literal ask), reproducibility row in appendix
- [ ] LSAC fix tested with a written outcome either way
- [ ] 12–40× claim substantiated or corrected in print
- [ ] docs/VERIFICATION_FINAL.md: zero mismatches
- [ ] `make test && make validate && make paper && make report` green from clean clone
- [ ] Advisors received the pre-read before submission

**The standard is unchanged since Jun 30: every number defensible, every negative
result reported. Now there is also no compute excuse.**
