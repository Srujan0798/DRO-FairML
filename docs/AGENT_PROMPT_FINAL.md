# Final prompt for your agent — paste verbatim

---

You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`,
picking this up after TASK F (the canonical reproducibility re-run) has
finished running in the background. This is the last step before submission.

I want you to be skeptical of `docs/FINAL_REPORT.md`'s checklist, not just
confirm it. A prior agent marked 12 ablations, a correctness audit, and a GPU
lane all done with no independent verification shown behind the checkmarks —
and separately, this project has caught three overstated claims already this
week (see `docs/AL_REVIEW.md` and its two supplements) that each reached a
committed file, and once reached a *built PDF*, before an independent pass
caught them. Assume the same could be true here until you've checked.

## 1. Confirm TASK F actually finished and verify it

```
python3 -c "import json; print(len(json.load(open('results/canonical_tau1_cosine.json'))))"
```

Should read 540. If it doesn't, the background job died early — check
`logs/task_f_repro.log` for the last line and figure out why before doing
anything else; do not proceed on a partial file.

If it's 540: run `experiments/verify_reproducibility.py`. Confirm, don't
assume, the predicted outcome: accuracy reproduces exactly, DP shifts by
~1e-7, IF goes from noise (~1e-11) to real values on the DP/COMBINED rows.
Write `results/task_f_verification_summary.md` with the actual measured
deltas. If anything is off by more than that, stop and report it — it would
mean the reproducibility story in `docs/MEMO_FOR_ADVISOR.md` section 6(b) is wrong.

## 2. Spot-check the self-reported checklist, don't just read it

Pick two items from `docs/FINAL_REPORT.md`'s Wave-1 ablation table that you
did NOT run yourself, and recompute their headline number directly from the
`results/*.json` they cite — not from the `.md` summary, from the raw JSON.
If your number matches, say so with the number. If it doesn't, that's the
most valuable thing you can find right now.

## 3. Confirm the paper and report actually contain the corrections

```
make paper && make report
pdftotext paper/main.pdf - | grep -c "3.7\|1.6"
pdftotext report/report.pdf - | grep -c "3.7\|1.6"
```

That checks the A4 corrected multiplier renders. If either grep returns 0,
the correction is in the source .tex but never made it into the built PDF
someone would actually submit — fix and rebuild.

## 4. Read docs/MEMO_FOR_ADVISOR.md once, end to end, as if you were Prof.
Manisha seeing it for the first time

Does it read as honest and precise, or does anything oversell? You have no
stake in the AL result being good — say plainly if any sentence in there
still reads like a stronger claim than the numbers support.

## 5. Final report

Write `docs/SUBMISSION_READY.md`: TASK F's actual verified deltas, the two
spot-checked ablation numbers with their source, PDF-rendering confirmation,
and an honest one-paragraph verdict — ready to submit, or not, and why.

pytest tests/ -q must be 101 passed or more before you finish. Commit and
push. Do not touch results/canonical_tau1.json or
results/utkface_canonical.json. Do not merge results/canonical_tau1_cosine.json
into the locked canonical file or switch any table over to it — that is
Prof. Manisha's decision (docs/MEMO_FOR_ADVISOR.md section 6b), not yours to make.
