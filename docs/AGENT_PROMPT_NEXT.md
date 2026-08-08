# Prompt for your agent — paste verbatim

---

You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
Read `docs/FINAL_REPORT.md` first — it claims a correctness audit, 12
ablations, a GPU validation lane, and the DRO-FAIR-AL work (tasks A–F) are
all complete. **Your job is to verify that claim, not trust it.** Treat
every number in that report as a claim under review until you've recomputed
it yourself from raw data — this project has a documented history of
overstated claims reaching the built PDF (see `docs/AL_REVIEW.md`,
`docs/AL_REVIEW_SUPPLEMENTARY.md`, `docs/AL_REVIEW_SUPPLEMENTARY_2.md` for
three prior examples of exactly this, all caught by independent review).

## 1. Check TASK F first (likely still running or just finished)

`experiments/run_task_f_repro.py` was launched to re-run the full canonical
grid under the corrected cosine k-NN metric, writing to
`results/canonical_tau1_cosine.json` (never touches the locked
`canonical_tau1.json`). Check `logs/task_f_repro.log` and the row count:

```
python3 -c "import json; print(len(json.load(open('results/canonical_tau1_cosine.json'))))"
```

- If still running (<540 rows): wait for it, do not interrupt it, do not
  touch `canonical_tau1.json`.
- Once it hits 540: run `experiments/verify_reproducibility.py`. The
  predicted outcome (stated in the memo and final report) is accuracy
  reproduces exactly, DP shifts by ~1e-7, and IF goes from noise (~1e-11)
  to real values (~0.045) on the DP/COMBINED rows. **Confirm this
  prediction holds** rather than assuming it does. If DP shifts by more
  than ~1e-5 anywhere, or any accuracy doesn't match, stop and report it —
  that would mean the reproducibility gap is not what we think it is.
- Write `results/task_f_verification_summary.md` with the actual deltas,
  not just a pass/fail.

## 2. Independently spot-check the three highest-leverage ablation claims

Don't re-verify all 12 — pick the ones the paper's headline depends on most,
and recompute from raw JSON rather than reading the summary `.md` files:

- **N1 (attack × radius match, Spearman ρ=+0.668, p=0.0065)** — this answers
  Kuldeep's original question and is cited as a standalone finding. Recompute
  the correlation yourself from `results/` (find the N1 result file via
  `grep -rl "N1" results/*.md`).
- **A4 (random vs adversarial corruption multiplier corrected to -3.7×–1.6×,
  down from a stated 12-40× claim)** — this is flagged in the final report as
  a claim the paper had to walk back. Confirm the corrected range is what's
  actually in `paper/sections/*.tex` now, not the old 12-40× number.
- **S (n=6→n=10 extension, "6 significance flips, all DP wins stay
  significant")** — recompute the Wilcoxon at n=10 for at least the Adult
  DP α=0.2 cell yourself and confirm the flip count.

## 3. Check the paper and report actually reflect all of this

`make paper && make report`, confirm both build with zero errors, then
`pdftotext` each PDF and grep for the corrected numbers (not just the old
ones) to confirm they render, not just compile — this project has had that
exact failure mode before (a significance star survived in a built PDF
after the underlying number was already fixed in source).

## 4. Report

Write `docs/FINAL_VERIFICATION.md`: what you confirmed, what you couldn't
verify without new experiments, and any discrepancy between
`docs/FINAL_REPORT.md`'s claims and what you actually recomputed. If
everything checks out, say so plainly and specifically — don't pad it. If
something doesn't check out, that's the most valuable thing you can find
right now, this close to submission.

Run `pytest tests/ -q` before and after any change; it must stay at
101 passed (or more, if you add verification tests — never fewer).
Commit with a clear message; do not force-push; do not touch
`results/canonical_tau1.json` or `results/utkface_canonical.json` under any
circumstance.
