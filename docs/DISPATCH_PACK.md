# DISPATCH PACK — copy-paste prompts for every remaining agent (2026-08-04 → Aug 10)

Companion to `docs/MASTER_PROTOCOL_AUG10.md` (the why). This file is the **what to paste**.

## READ FIRST — state as of this writing

- Locked science (**never rewrite**): `results/canonical_tau1.json` (540 rows),
  `results/utkface_canonical.json` (90 REAL rows). Extensions APPEND only.
- Shared driver: `experiments/run_ablation_parallel.py` — refuses to write locked files,
  resume-safe, atomic parent-only merge, provenance stamped in parent.
- **PERF FIX LANDED (`42180a1`)**: the pool was dead (`ctx.get_start_agent()`
  AttributeError before the try block) so every ablation ran sequential. Fixed and
  verified: 4 configs in 13 s wall-clock vs ~52 s. **Every agent must now launch with
  `ABLATION_WORKERS=10`.** Any run already going at workers=1 should be killed and
  relaunched — it is resume-safe, nothing is lost.
- Done: N4 (IF@α=0.3 formalized), paper assembly (figures/tables/appendices wired).
- Running/queued: A1, A2, A3, A4, A5, N5.
- Throughput budget: ~13 s/config sequential, ~1.3 s/config effective at 10 workers.

## GLOBAL RULES — paste into every agent prompt

```
NON-NEGOTIABLE RULES:
1. NEVER write results/canonical_tau1.json or results/utkface_canonical.json. All new
   work goes to a NEW results/*.json. The driver enforces this; do not bypass it.
2. Launch with ABLATION_WORKERS=10 (pool fix 42180a1 verified). Never two writers on
   one JSON file. Check `ps aux | grep run_` before starting.
3. Full provenance on every row: tau, k_inner, epochs, pgd_steps, seed,
   n_seeds_planned, radii_mode, lambda_init, lr_lambda, coordinated, corruptor_type,
   attack_k — plus whatever your ablation varies.
4. `python3 -m pytest tests/ -q` green before every commit (currently 65 passing).
5. Report negative results AS negative. The τ=100 episode is the standing lesson:
   an honest null beats a flattering artifact. Never tune until it wins.
6. Every number that reaches a doc/paper must be recomputable from a committed
   results/*.json. If you cannot trace it, delete it.
7. Read docs/reference/ARCHIVE_POLICY.md before deleting or moving ANY file. Grep for
   references first — that check is what a deletion once broke (generate_figures.py).
8. Deliverable = data JSON + a summary .md + the paper/report text it feeds. Data alone
   is not done.
```

---

# WAVE 1 — relaunch at 10 workers (in flight)

## AGENT A1 — kNN ablation k ∈ {5,15}
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/DISPATCH_PACK.md GLOBAL RULES
and docs/MASTER_PROTOCOL_AUG10.md Part 1 row 5.

Kuldeep, Jun 9, verbatim: "For if attack we have to do ablation study for different k
5,10,15". k=10 already exists inside the locked canonical — do NOT re-run it.

Run via experiments/run_ablation_parallel.py with ABLATION_WORKERS=10:
  attack='if', attack_k ∈ {5,15}, 3 datasets × 5 alphas × 6 seeds × 2 methods
  = 360 configs → results/knn_ablation.json
Everything else canonical: tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0.

Deliverable results/knn_ablation_summary.md: per (dataset, α), DP/IF/acc at k=5/10/15
(pull k=10 rows from canonical for the comparison, read-only), plus paired Wilcoxon
k=5 vs k=10 and k=15 vs k=10 on BOTH the IF-violation metric and DP.
Answer the exact question in one sentence: does the IF attack's strength depend on k?
Then write the appendix paragraph for paper/sections/ (hand to Agent I2).
Expected ~8 min at 10 workers. Commit data + summary + appendix text.
```

## AGENT A2 — τ ablation {10,100}, clean this time
```
Same working dir + GLOBAL RULES. Kuldeep, Jun 9: "we fix tau for all alpha. Here we can
use different tau for ablation study."

The previous τ ablation was DROPPED because it confounded τ with k_inner and had no
LSAC. Do not repeat that: k_inner=10 EVERYWHERE, all three datasets.

  tau ∈ {10,100}, attack='dp', 3 datasets × 5 alphas × 6 seeds × 2 methods
  = 360 configs → results/tau_ablation.json   (τ=1 = canonical, do not re-run)

Deliverable results/tau_ablation_summary.md: the τ=100 artifact demonstrated CLEANLY on
all three datasets — DP(τ=1) vs DP(τ=10) vs DP(τ=100) per α, showing the flip
(DRO loses at τ=100 → DRO wins at τ=1). This is the paper's MOTIVATING ablation: it is
the evidence that the original "DRO is fragile" conclusion was a temperature artifact,
and it is currently asserted in prose with no committed data behind it. Also produce the
figure (τ on x or as series, DP on y, per dataset). ~8 min. Commit data + summary + figure.
```

## AGENT A3 — λ/lr grid (pathology-aware)
```
Same working dir + GLOBAL RULES. Kuldeep Q1, Jun 9: "try different initial value of
lemdas, learning rates or similarly hyper parameters tuning to relax accuracy and tight dp".

The old 720-cell grid died on a λ0=1.0 config that ran 17.9 HOURS. EXCLUDE λ0=1.0.
  lambda_init ∈ {0.0, 0.01, 0.1} × lr_lambda ∈ {0.001, 0.005}
  Adult, attack='dp', α ∈ {0.2, 0.3}, 6 seeds, DRO only = 72 configs
  → results/lambda_grid.json
If the fast path holds, ALSO extend to α ∈ {0.4} and dataset credit (+72) — cheap now.

Deliverable results/lambda_grid_summary.md + paper/sections/appendix_q1_lambda.tex
refreshed. Two questions to answer explicitly:
 (a) does any (λ_init, lr_λ) beat default (0.0, 5e-3) on DP without accuracy loss?
 (b) does ANY cell rescue α=0.3 accuracy above the Adult constant predictor 0.7521?
Old partial data said no to (b). Confirm or refute with the full grid and SAY SO.
~3 min. Commit.
```

## AGENT A4 — random vs adversarial (backs the quoted 12–40×)
```
Same working dir + GLOBAL RULES. NOTE: a sequential run is in flight — kill it and
relaunch with ABLATION_WORKERS=10 (resume-safe, keeps completed rows).

  corruptor_type ∈ {'random','adversarial'}, attack='dp', 3 datasets ×
  α ∈ {0.1,0.2} × 6 seeds × 2 methods = 144 configs
  → results/random_vs_adversarial.json
Extend to α ∈ {0.3,0.4} if fast (+144) — gives the full corruption curve.

THE POINT: on Jun 16 we told Kuldeep adversarial PGD is "12-40x stronger than random".
That number came from pre-canonical data that no longer exists. This run either
SUBSTANTIATES it or CORRECTS it, and the paper prints whatever is true. Deliverable
results/random_vs_adversarial_summary.md: ΔDP(adversarial)/ΔDP(random) per dataset/α
with the multiplier and CIs, plus the figure. If the real multiplier is not 12-40×,
say the real number and flag that the earlier chat figure was superseded.
~4 min. Commit.
```

## AGENT A5 — empirical radii (Q5)
```
Same working dir + GLOBAL RULES. Kuldeep, Jun 9: "In paper if attack is known then we
can use this approximation according to attack."

  radii_mode='empirical', coordinated=True, attack='dp',
  Adult + Credit × 5 alphas × 6 seeds × 2 methods = 120 configs
  → results/empirical_radii.json
Compare against the canonical uniform rows (read-only).

Deliverable results/empirical_radii_summary.md + paper/sections/appendix_q5_empirical.tex.
The question: does attack-aware radius calibration improve DRO under coordinated
corruption vs the uniform closed form? NOTE for correctness: verify empirical mode is
actually ACTIVE — an earlier version silently no-op'd under coordinated=False, and the
uniform closed form was found to be dead code because a_val was always passed. Assert
the radii actually differ between modes (log rho per run) before trusting any result.
~3 min. Commit.
```

## AGENT N5 — K_inner ablation {5,20} (closes Q10)
```
Same working dir + GLOBAL RULES. Kuldeep Q10, Jun 9: is K_inner=5 acceptable vs 10?
Answer it with committed data instead of the old deleted "virtually identical" claim.

  k_inner ∈ {5,20} (10 = canonical, do not re-run), DRO only, attack='dp',
  3 datasets × 5 alphas × 6 seeds = 180 configs → results/kinner_ablation.json

Deliverable results/kinner_ablation_summary.md + one appendix table: does K_inner beyond
5 change DP/IF/accuracy materially? Include wall-clock per config so the
compute-vs-benefit tradeoff is visible. ~4 min. Commit.
```

## AGENT G — flair2 unlock (gates Wave 2 server work)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_PROTOCOL_AUG10.md
Part 2. scripts/flair2_unlock.sh may already exist — check before rewriting.

flair2 state: SSH key auth WORKS (`ssh flair2`), 2× NVIDIA L40S 46 GB, driver 570
(Supin fixed it 2026-08-04), code + real UTKFace features already at
/data/srujan.sai/DRO-FairML-run/. Missing: torch. The node is behind an SSL-intercepting
firewall so pip CANNOT reach PyPI — offline wheel install is the only path.

A ~830 MB wheelhouse already exists locally (torch 2.6.0 + torchvision + sci stack) but
is MISSING the 14 nvidia-cu12 CUDA wheels (~1.7 GB). Download them ON THE MAC (exact
pins in Part 2 of the protocol — torch 2.6.0 requires cu12 12.4.127 / cudnn 9.1.0.70 /
triton 3.2.0), rsync the wheelhouse to flair2, create venv_gpu, install with
--no-index --find-links.

Campus wifi throttled this to ~178 kB/s earlier. Use ethernet, or run it overnight, or
resume with rsync (it is restartable). Do not kill other work waiting on it.

GATE — do not report success until this prints exactly "True 2":
  ssh flair2 '/data/srujan.sai/DRO-FairML-run/venv_gpu/bin/python -c
  "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"'
Then commit the unlock script + a short docs note. Wave 2 Agent U depends on this gate.
```

---

# WAVE 1.5 — the deeper asks (start as Wave 1 frees cores)

## AGENT N1 — attack strength × radius (Kuldeep's FIRST question, never answered)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES + protocol Part 1B D1.

Kuldeep, May 29, verbatim — the first technical question he ever asked, still unanswered
14 months of project-time later: "At lower corruption levels (α=0.1): DRO does not
significantly outperform Naive — the attack is too weak to differentiate. Does the attack
affect the radius? ... if the attack is too weak, then DRO would perform well? specially
at α=0.1."

He is asking whether DRO's advantage is a function of the MATCH between attack strength
and radius calibration. Answer it with two arms.

ARM A — vary attack strength at fixed α:
  pgd_steps ∈ {5, 50} (canonical 20), attack='dp', 3 datasets × α ∈ {0.1,0.2} ×
  6 seeds × 2 methods = 144 configs → results/attack_strength.json
  CRITICAL: also record MEASURED attack effectiveness per run (the ΔDP the corruption
  itself induces on the training labels, pre-training). Strength must be measured, not
  assumed — that is the whole point of his question. Add it as a provenance field.

ARM B — vary radius at fixed attack: add a `radii_scale` kwarg to
  DroFairTrainer._compute_radii (multiply rho_dp/rho_if; provenance-recorded; default
  1.0 so nothing else changes; add a unit test). radii_scale ∈ {0.5, 2.0},
  3 datasets × dp × 5 alphas × 6 seeds, DRO only = 180 configs
  → results/radius_sensitivity.json

Deliverable results/attack_radius_summary.md + ONE figure that answers him directly:
DRO advantage (Naive DP − DRO DP) vs measured attack strength, and vs radius scale.
If the advantage peaks when radius matches true corruption, that is a genuine finding
and belongs in the paper. If it is flat, say that. ~6 min compute at 10 workers; the
radii_scale code change + test is the real work. Commit code+test, data, summary, figure.
```

## AGENT N2 — high-α rescue, Kuldeep's exact 3-step protocol
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES + protocol Part 1B D2.

Kuldeep, Jun 16, dictated a procedure we never executed: "Different tau value 1st if not
improving then change learning rates for lamda or something else check loss convergence
plots and choose according to it on validation set". Execute it LITERALLY, IN ORDER.

STEP 1 — per-α tau: tau ∈ {2,5,20}, Adult, dp, α ∈ {0.3,0.4}, 6 seeds, 2 methods
  = 72 configs → results/high_alpha_tau.json
STEP 2 — if Step 1 does not lift accuracy above 0.7521: lr_lambda ∈ {0.01} arm at the
  same α (24 configs, same file). Coordinate with A3 to avoid duplicate cells.
STEP 3 — THE PART NEVER DONE, and the reason he asked: CONVERGENCE DIAGNOSTICS.
  Instrument DroFairTrainer and NaiveFairTrainer to record per-epoch train loss, val
  loss, and val accuracy into a `history` list, dumped alongside each result row (guard
  it behind a flag so canonical reruns are unaffected; add a unit test).
  Then add the arm that tests the real hypothesis: epochs=200 with validation-based
  EARLY STOPPING (patience 20) at α ∈ {0.3,0.4}, Adult, dp, 6 seeds, 2 methods
  = 24 configs.
  HYPOTHESIS: 60 fixed epochs UNDERFITS at high corruption. If val-selected longer
  training lifts accuracy above the constant predictor (Adult 0.7521), THE DEFENSIBLE
  REGIME EXTENDS BEYOND α=0.2 — that is a headline-level upgrade to the paper's main
  claim. If it does not, the α≥0.3 limitation finally has the convergence evidence
  Kuldeep asked for instead of an assertion.

Deliverable: results/high_alpha_summary.md + the literal artifact he requested —
train/val loss-vs-epoch convergence plots per τ per α — + paper text that either extends
the defensible regime or closes it with evidence. Commit code+tests, data, plots, text.
```

## AGENT N3 — COMPAS + German Credit ("Adult etc")
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES + protocol Part 1B D3.

Manisha, May 19: "see the performance of DRO on Adult etc". A fairness paper with no
COMPAS is conspicuous to any reviewer.

1. Add two loaders to src/data/datasets.py following the existing get_dataset() contract
   (returns X/y/a train+val+test triples + display name):
   - COMPAS (ProPublica recidivism; protected = race African-American vs Caucasian;
     label = two_year_recid)
   - German Credit (UCI statlog; protected = sex, or age<25; label = good/bad credit)
   Add both to data/download_data.sh (public URLs, checksum if practical) and write unit
   tests: shapes, no NaN, protected-attribute balance, train/test disjoint, and the
   constant-predictor baseline accuracy for each (needed for the α≥0.3 comparison).
2. Full canonical protocol on both: 2 datasets × 3 attacks × 5 α × 6 seeds × 2 methods
   = 360 configs → results/extended_datasets.json  (NEW file, never canonical).

Deliverable: extended main-results table spanning FIVE tabular datasets + UTKFace, and
an honest replication verdict: does the Adult/Credit pattern (DRO better on DP at α≤0.2)
REPLICATE on COMPAS and German? If it does, the paper's claim generalizes and that is a
major strengthening. If it does not — especially on German, which is small (1000 rows)
and noisy — report the scope limit plainly. A stated scope beats an overclaim.
~15 min compute; loaders+tests are the real work. Commit loaders+tests, then data,
then summary.
```

---

# WAVE 2 — upgrades the hardware now permits

## AGENT S — n=6 → n=10 seeds (APPEND ONLY)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES.

Kuldeep Q9, Jun 9: "6 seeds now, or is 3 acceptable" / "or push for more?" At n=6 the
minimum attainable Wilcoxon p is 0.0156; at n=10 it drops to ~0.001. The project owner
has GREENLIT extending the canonical grids — APPEND ONLY, existing rows are never
modified or recomputed.

TABULAR: seeds 6-9 → 3 datasets × 3 attacks × 5 α × 4 seeds × 2 methods = 360 configs,
APPENDED to results/canonical_tau1.json. This is the ONE sanctioned exception to the
locked-file rule and it requires: resume-safe missing-key enumeration, single writer,
atomic write, and a hard post-check that the file has exactly 900 rows with the original
540 byte-identical. If any pre-existing row changes, ABORT and restore.
UTKFACE: seeds 6-9 → 60 rows appended to results/utkface_canonical.json (150 total).

THEN regenerate EVERYTHING downstream at n=10: Wilcoxon (all cells), summary CSVs,
tables, figures, both PDFs — and update every "n=6" caption and claim in paper, report,
STATUS.md, and docs to n=10. If any cell FLIPS significance at n=10, the paper reports
the n=10 truth and the change is called out explicitly in the summary.
~50 min at 10 workers. Commit data first, artifacts second, prose third.
```

## AGENT U — UTKFace on the server + multi-group + pixel-PGD
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. REQUIRES Agent G's gate ("True 2").
Read GLOBAL RULES + protocol Part 1B D7/D8.

(a) THE LITERAL ASK — Manisha, May 19 task 2: "Set up an experiment for the UTKFace
    dataset IN THE SERVER". She asked again Jun 19 ("Are you guys able to access
    flair2??"). Run the identical UTKFace canonical grid on flair2 with --device cuda
    (scripts/deploy_utkface_flair2.sh exists) → results/utkface_flair2.json.
    Same seeds as the Mac MPS run ⇒ near-identical numbers ⇒ a REPRODUCIBILITY appendix
    row (CPU/MPS vs CUDA agreement), and the May-19 task is finally, literally true.
(b) MULTI-GROUP (D8): UTKFace race is currently collapsed to binary White/non-White.
    src/evaluation/metrics.py ALREADY computes max-min DP for >2 groups. Re-run with
    race as 5 GROUPS: 5 α × 6 seeds × 2 methods × dp = 60 configs
    → results/utkface_multigroup.json. Multi-group fairness under adversarial corruption
    is strictly stronger than binary and is likely novel at this scope.
(c) PIXEL-SPACE PGD (stretch — only a 46 GB GPU can do this): attack raw UTKFace images
    through ResNet18 instead of the cached feature vectors. src/corruption/image_pgd.py
    exists in git history (recover per ARCHIVE_POLICY: git log --all --
    src/corruption/image_pgd.py). Scope: α ∈ {0.1,0.2}, 6 seeds, dp, 2 methods.
    Gives the paper a "feature-space vs pixel-space attack" section.
(d) CelebA second modality — only if (a)-(c) land by Day 3. Cut without guilt.

Provenance: every row must be tagged with device and data_provenance=REAL. NEVER let
synthetic features (run_utkface.py::_make_synthetic_utkface) be reported as real — the
loader already rejects them, keep it that way. Commit each sub-deliverable separately.
```

## AGENT L2 — LSAC degeneracy: test the fix (hypothesis, not tuning)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES +
docs/LSAC_DEGENERACY.md.

Kuldeep Q4, Jun 9, still only half-answered: the LSAC α=0 anomaly. We DIAGNOSED it —
LSAC is ~90/10 imbalanced, the model collapses to the majority class, DP freezes at
0.1827 for α ∈ {0.2,0.3,0.4}, accuracy pins at the 0.9016 constant-predictor baseline,
and DRO loses 0/6 seeds. We never TESTED the fix the diagnosis implies.

Test it: per-group radius clamping (cap rho_dp[j] so the tiny minority group cannot blow
up) and/or radii_mode='empirical', on LSAC, attack='dp', 5 α × 6 seeds × 2 methods
= 60 configs → results/lsac_radii_fix.json. Coordinate with N1's radii_scale work to
avoid conflicting edits to _compute_radii.

THE RULE, stated because this is exactly where the temptation lives: this is HYPOTHESIS
TESTING, not tuning-until-it-wins. Pick the clamp on principle (documented before you
run), run once, report.
 - If the fix un-degenerates LSAC (accuracy moves off the 0.9016 pin, DP unfreezes
   across α), the paper UPGRADES LSAC from "degenerate, excluded" to "recovered by
   attack-aware radius calibration" — a genuine methodological contribution.
 - If it does not, the limitation stands WITH EVIDENCE instead of a hypothesis.
Both outcomes ship. Only the untested state does not. Commit.
```

---

# WAVE 3 — integration and the gate

## AGENT I2 — full integration
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read GLOBAL RULES.

Every Wave 1/1.5/2 result must reach the paper and report, or be explicitly cut in
writing. Checklist:
- Main results table spans all datasets that landed (3 tabular + COMPAS/German if N3
  landed + UTKFace), at the final n (10 if Agent S landed, else 6).
- Ablation sections: τ (A2), kNN (A1), K_inner (N5), λ grid (A3), empirical radii (A5).
- New-finding sections: attack×radius (N1), high-α convergence (N2), random-vs-adv
  multiplier (A4), LSAC fix outcome (L2), UTKFace server/multi-group/pixel (U).
- D6: EVERY accuracy figure carries the 0.78 reference line AND the per-dataset
  constant-predictor line (Adult 0.7521, Credit 0.7788, LSAC 0.9016, + new datasets
  computed from data — never hardcoded; the old hardcoded 0.752 bug applied Adult's
  baseline to every dataset).
- Report and paper tell the SAME story with the SAME numbers at the SAME n.
- Regenerate all figures/tables from final data; `make paper && make report` build clean;
  visually confirm every figure and table actually renders in the PDF (a paper that
  builds but shows nothing was the exact failure found on Aug 4).
Commit in logical chunks.
```

## AGENT V2 — final verification gate (Kuldeep's standing request)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. You are the adversarial checker.
Assume every claim is wrong until you recompute it from a committed results/*.json.

Kuldeep, Jun 30: "After drafting the reply, could you please verify all the claims?
Sometimes AI tends to make claims just to make the results appear correct." That request
has already caught real errors twice in this project (mislabelled IF plots; a hidden
LSAC negative). It is the gate.

1. Recompute EVERY mean, win-count, and p-value appearing in: paper/, report/,
   README.md, STATUS.md, docs/*.md, and every results/*_summary.md — directly from the
   JSON. Include the random-vs-adversarial multiplier and every ablation sentence.
2. Verify every caption's stated n matches the data behind it.
3. Provenance sweep: every results file uniform in tau/k_inner/epochs where it should be;
   no synthetic UTKFace rows anywhere; no orphaned//stale file feeding any table.
4. grep for hardcoded numbers in figure generators and .tex (the 0.752 class of bug).
5. Run: make test, make validate, make paper, make report.
6. Produce docs/VERIFICATION_FINAL.md — a table of claim → source file → recomputed
   value → MATCH / MISMATCH.

ZERO unresolved MISMATCH rows may ship. Anything that cannot be traced gets deleted from
the document, not explained away.
```

## AGENT R — advisor pre-read (Day 6)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML.

Produce docs/ADVISOR_PREREAD.md for Prof. Manisha Padala and Kuldeep — sent BEFORE
submission so they can flag anything.

Structure:
1. What was asked and what was delivered — walk their own asks (Manisha May 19 tasks 1&2,
   Jun 2 redo; Kuldeep's Q1-Q13, the May-29 radius question, the Jun-16 tau/lambda/
   convergence protocol) and state delivered / not delivered with a one-line pointer.
2. Headline results at final n, with the honest caveats intact: LSAC/DP degeneracy (and
   whether L2 recovered it), the α≥0.3 constant-predictor bound (Adult+Credit; LSAC is
   pinned AT baseline, not below), IF MIXED under DP but DRO-favourable on the IF metric.
3. Corrections to anything previously reported wrong (docs/KULDEEP_CORRECTION.md).
4. What was cut and why — explicitly, not silently.
5. Attach: paper PDF, report PDF, and the key figures.
Tone: factual, no spin, no defensiveness. They are the last check before submission.
```

---

# EXECUTION ORDER

```
NOW      relaunch A1 A2 A3 A4 A5 N5 at ABLATION_WORKERS=10   (~30 min total, all parallel)
         G (flair2 torch) starts alongside — slow network, needs the runway
DAY 1-2  N1 N2 N3   (code changes + runs; N2 and N1 both touch trainers — serialize
                     the trainer edits, then run in parallel)
DAY 2-3  S (n=10, append-only)  ·  U (needs G's gate)  ·  L2 (coordinate _compute_radii
                     edits with N1)
DAY 4-5  I2 integration → V2 verification gate
DAY 6    R advisor pre-read; fix whatever they flag
DAY 7    submit
```

**Conflict map — the only real collision risks:**
- `src/training/dro_fair.py::_compute_radii` — N1 (radii_scale), L2 (clamping), A5
  (empirical). **Serialize these three edits**, then run experiments in parallel.
- Trainer `history` logging — N2 only.
- `results/canonical_tau1.json` — Agent S only, append-only, everyone else read-only.
- Paper `.tex` — Agent I2 owns integration; A1/A2/A3/A5/N2 hand it text, don't edit
  main.tex concurrently.
