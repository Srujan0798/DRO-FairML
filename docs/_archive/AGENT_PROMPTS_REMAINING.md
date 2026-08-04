# Copy-paste prompts — REMAINING work (2026-07-20)

Repo: `/Users/srujansai/Desktop/DRO-FairML`. Full context in `docs/MASTER_DISPATCH.md` (see §0 blockers and §0.5 progress).

State of play: 8 commits landed, tree clean, 62 tests pass, both PDFs rebuilt from the
canonical file. The canonical is **360 rows (DP + Combined attacks only)** — the IF-attack
third was never generated because the IF metric was degenerate. Agent A has since FIXED the
metric. Four items remain. G1 needs a cluster; G2/G3/G4 do not and can run now in parallel.

---

## AGENT G1 — Run the 180 IF-attack rows on a cluster (needs compute, not this laptop)

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
§0.5 first.

CONTEXT: the canonical grid results/canonical_tau1.json currently holds 360 rows —
the DP and Combined attacks only (3 datasets x 5 alphas x 6 seeds x 2 methods x 2
attacks). The IF-attack third (180 rows) was never generated: the individual-fairness
metric was degenerate (~1e-10 everywhere) in the original runs. That metric is now
FIXED in src/evaluation/metrics.py (cosine-based; verified constant predictor -> 0.0,
unfair predictor -> 0.28 on real Adult), and the IF attack gradient in
src/corruption/adversarial.py is aligned to the same objective and the same global
k-NN graph used at eval.

YOUR TASK: generate the missing 180 IF-attack rows and rebuild everything downstream.

1. Verify the metric fix is present before spending compute:
     python3 -m pytest tests/test_metrics.py -q      # must pass
   and confirm src/evaluation/metrics.py compute_if_violation is cosine-based (not the
   old relu(|h_i-h_j| - d_ij - gamma) that saturated to zero).

2. Launch scripts/run_if_rerun_cluster.sh on a machine with real compute. Each
   (dataset, alpha, method) config is ~3-6 min on CPU; the full sweep is ~15 h single-CPU.
   GPU is NOT required (this is the tabular FairnessTargetedPGD task). The config is
   pinned inside the script: tau=1.0, k_inner=10, seeds=6, attack=if. run_fairness_pgd.py
   is resume-safe — it appends only missing (dataset,alpha,seed,attack,method) keys to
   canonical_tau1.json, so the job can be relaunched freely and will NOT touch the 360
   existing DP/Combined rows.

3. When the run finishes, confirm:
     python3 -c "import json,collections; d=json.load(open('results/canonical_tau1.json')); print(len(d), collections.Counter(r['attack'] for r in d))"
   Expect 540 total, {dp:180, if:180, combined:180}. Confirm IF is non-degenerate:
     python3 -c "import json; d=json.load(open('results/canonical_tau1.json')); print(max(abs(r['if_clean']) for r in d if r['attack']=='if'))"
   This must be >> 1e-6 (the old value was 4.7e-10).

4. Regenerate everything downstream from the now-540-row canonical (the cluster script
   does this automatically at the end; verify it ran): Wilcoxon across all 45 cells,
   summary CSVs, all figures, both PDFs via tectonic (make paper && make report).

5. Report the new IF Wilcoxon results honestly — including any cell where DRO loses.
   Do NOT tune anything to make IF look better. The IF plots sent to the collaborator on
   Jun 30 were mislabeled DP data; the correction (Agent G4) depends on these being the
   first REAL IF numbers the project has ever produced.

COMMIT the updated canonical_tau1.json and all regenerated artifacts.
```

---

## AGENT G2 — Resolve UTKFace: it is unverified and referenced in 7 .tex files

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
§0 first.

THE PROBLEM: docs/UTKFACE_RESULTS.md claims a "GPU Server Run — 23,705 images,
ResNet18 features, 5 seeds per alpha" dated May 29. The rest of the repo says GPU access
on flair2.iitgn.ac.in was NEVER granted (the email to supin.gopi was never answered) and
only a 2-row CPU smoke test exists. experiments/run_utkface.py:52-105 silently
substitutes _make_synthetic_utkface (random Gaussian features) whenever the real dataset
is missing, tagging results only via dname='UTKFace (synthetic)'. So those "results" are
almost certainly synthetic noise presented as a real GPU run.

This is not confined to one doc. grep shows UTKFace is referenced in SEVEN paper/report
files:
  paper/main.tex, paper/sections/{discussion,conclusion,appendix_q5_empirical,
  experimental_setup,results}.tex, report/report.tex
and three derived figure pairs exist:
  figures/fig10_utkface_curves.{pdf,png}
  figures/fig_utkface_dp_comparison.{pdf,png}
  figures/fig_utkface_tradeoff.{pdf,png}

YOUR TASK: establish the truth and make the repo consistent with it.

1. Determine whether the UTKFACE_RESULTS.md numbers came from real images or from
   _make_synthetic_utkface. Check: the results file behind the doc (grep for utkface in
   results/), whether any real UTKFace image data ever existed on disk, and the dname
   field on those rows. Look at git history for when/how the file was produced.

2. If they are synthetic (most likely):
   - Retitle docs/UTKFACE_RESULTS.md to make it unmissable, e.g. "UTKFace — SYNTHETIC
     SMOKE TEST ONLY (NOT REAL RESULTS)", with a top banner explaining these are random
     Gaussian features, GPU access was never granted, and no real image experiment was
     ever run. Move it to docs/_archive/ if that's where superseded material lives.
   - Delete the three derived figure pairs listed above (they visualize synthetic data).
   - Remove or clearly caveat every UTKFace claim in the 7 .tex files. UTKFace should be
     described as future/blocked work, not as a completed experiment. Rebuild both PDFs
     (make paper && make report) and confirm no figure include is left dangling.

3. If by some chance they ARE real: document the provenance (where the images came from,
   the run command, the results file) so the claim is defensible. Then leave the figures
   but ensure the .tex references cite the provenance.

4. Separately: docs/EMAIL_TO_SUPIN_GOPI_DRAFT.txt is the GPU-access request, drafted but
   apparently never sent, and it is signed "Rapuru Ganesh (23110271)". Decide with the
   human whether UTKFace is being formally dropped from scope (in which case note it in
   STATUS.md) or the email should still go out. Do not send it yourself.

CONSTRAINT: run make paper && make report at the end; both must build. 62 tests must
still pass. Report exactly what you found (real vs synthetic) and every file you changed.
```

---

## AGENT G3 — Rewrite STATUS.md to the 360-row truth

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
(all of §0 and §0.5) first — it has the verified numbers.

THE PROBLEM: STATUS.md calls itself the project's "single source of truth" but is stale
and now actively contradicts reality. It still says the canonical grid is "540 rows —
running", references an "orchestrator" that auto-regenerates "at 540", and lists work as
in-progress that is long since done or dropped. It was last updated by an old commit and
none of the six agents touched it.

YOUR TASK: rewrite STATUS.md so it matches the committed state of the repo. Verify every
line against the actual files — do not copy claims from the old STATUS.md or from
MASTER_DISPATCH §0 without re-checking.

Ground truth to encode (verify each yourself):
- Canonical grid is 360 rows in results/canonical_tau1.json: DP + Combined attacks only,
  3 datasets x 5 alphas x 6 seeds x 2 methods x 2 attacks, all tau=1.0, k_inner=10.
  (Verify: python3 -c "import json,collections; d=json.load(open('results/canonical_tau1.json')); print(len(d), collections.Counter(r['attack'] for r in d))")
- The IF-attack third (180 rows) is NOT done — it is cluster-blocked (Agent G1). IF cells
  in all tables currently read 0.0000 (degenerate pre-fix metric). The metric is fixed in
  code but the 180 rows have not been regenerated.
- Verified positive results (n=6): Adult DP wins every alpha (p<=0.031); Adult Combined
  6/6 every alpha (p=0.016); Credit all attacks ~14/15 cells p<0.05; LSAC Combined
  p=0.016 at alpha 0.1/0.3/0.4.
- Verified negatives: LSAC/DP is DEGENERATE (DRO 0/6 at every alpha, p=1.0, DP frozen at
  0.1827, accuracy pinned to the constant-predictor baseline 0.9016) — see
  docs/LSAC_DEGENERACY.md. alpha>=0.3 is below the constant-predictor baseline on every
  dataset (adult 0.7521, credit 0.7788, lsac 0.9016), so no method claim is made there.
- Defensible regime: alpha <= 0.2 on Adult and Credit.
- Ablations: adjudicated in docs/ABLATION_STATUS_REPORT.md (tau/lambda/random-vs-adv
  dropped with reasons; kNN retracted). Reflect that, don't re-open them.
- UTKFace: blocked / unverified (see Agent G2's resolution — coordinate on final wording).
- Both PDFs build from canonical via make paper / make report.

Requirements:
- Date it 2026-07-20.
- Remove every reference to a "540-row run in progress", the "orchestrator auto-regen",
  and any "pending/running" item that is actually done or dropped.
- Add a short "What remains" section listing ONLY: the IF cluster run (G1) and the
  UTKFace decision (G2).
- Keep it to one screen. It is a status doc, not a report.
```

---

## AGENT G4 — Draft the honest correction note to Kuldeep

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
§0 and the chat log at docs/chat/gchat_raw_export.md first.

CONTEXT: on Jun 30 the collaborator Kuldeep explicitly asked, after learning an AI agent
was drafting the replies: "After drafting the reply, could you please verify all the
claims? Sometimes AI tends to make claims just to make the results appear correct."
Since then, verification against the completed data has found that three things reported
to him were wrong. He needs an honest correction. This is the highest-trust deliverable
in the project — it must be scrupulously accurate and must not spin.

YOUR TASK: write docs/KULDEEP_CORRECTION.md — a plain, short, honest note he can read in
two minutes. Cover exactly these three corrections, each with the specific prior message
being corrected and the verified current number:

1. IF PLOTS WERE MISLABELED. On Jun 30 (5:47pm and 5:59pm) plots named adult_if_*.pdf
   were sent as "individual fairness", and the reply quoted "IF violation: DRO = 0.0195
   vs Naive = 0.0177". Those numbers came from the DP column, not IF. The IF metric was
   degenerate (identically ~0 across all rows) due to a threshold-calibration bug, so the
   project had no valid IF results at that time. The metric is now fixed (cosine-based)
   and the real IF experiment is being re-run; honest IF numbers will follow from that
   run (Agent G1). Until then, make NO IF claim.

2. LSAC WAS REPORTED AS "PENDING" BUT IS COMPLETE AND NEGATIVE. LSAC under the DP attack:
   DRO loses to Naive at every alpha, 0/6 seeds, p=1.0. It is in fact a degenerate run —
   the model collapses to the constant predictor (DP frozen at 0.1827 for alpha
   0.2/0.3/0.4; accuracy pinned to the 0.9016 majority-class baseline). See
   docs/LSAC_DEGENERACY.md. LSAC/Combined, by contrast, is a genuine win (p=0.016 at
   alpha 0.1/0.3/0.4). Do not hide the DP loss.

3. THE alpha>=0.3 REGIME IS BELOW THE CONSTANT PREDICTOR. On every dataset, at alpha>=0.3
   both DRO and Naive drop below constant-predictor accuracy (adult 0.7521, credit
   0.7788, lsac 0.9016), so the "advantage grows with alpha" framing is empty there — the
   growth is inside a regime where both methods are useless. The defensible claim is
   scoped to alpha <= 0.2.

Then state clearly WHAT IS SOLID and can be led with (verified n=6): Adult DP wins every
alpha up to the defensible bound (0.1491/0.1426 ... 0.2452/0.2334 at alpha 0.2, all
p<=0.031); Adult Combined 6/6; Credit all three attacks p<0.05 at nearly every cell.

Tone: factual, brief, no defensiveness, no spin. Lead with the corrections, not the wins.
Every number must be reproducible from results/canonical_tau1.json — if you cannot trace
it, do not write it. Do NOT send anything; produce the file for the human to review and
send.
```
