# SUBMISSION_READY — Final verdict

**Date:** 2026-08-10 · **Branch:** main · **Tests:** 101 passed
**Paper:** `paper/main.pdf` (393 KiB) · **Report:** `report/report.pdf` (320 KiB)

---

## 1. TASK F — verified deltas (independent recompute)

Committed data: `results/canonical_tau1.json` (locked 540) vs
`results/canonical_tau1_cosine.json` (re-run 540), paired by
(dataset, alpha, seed, method, attack). Full detail:
`results/task_f_verification_summary.md`.

| Attack | n | acc max | DP max | Status |
|---|---|---|---|---|
| dp | 180 | 0.0028 | 0.0031 | stable |
| if | 180 | 0.0053 | 0.0082 | stable |
| combined | 180 | **0.0746** | **0.0850** | drift at α≥0.2 |

- IF column: noise in 360/540 rows before → real measurement in 540/540 after. ✅ Fix works.
- **Two n=6 significance flips** (the previous "paper claims robust" verdict missed these):

| Cell | Locked | Re-run |
|---|---|---|
| adult / dp / α=0.1 | 5/6, p=0.0312, sig | **4/6, p=0.1094, not sig** |
| lsac / combined / α=0.1 | 6/6, p=0.0156, sig | **4/6, p=0.0781, not sig** |

All other cells keep significance. Verified with the repo's own
`compute_canonical_wilcoxon.py::compute_wilcoxon`.

## 2. Spot-checked ablation numbers (from raw JSON, not summaries)

- **A4 (random vs adversarial, 144 rows):** multiplier range −3.72× to +1.57×,
  median 0.66×. Matches report's corrected −3.7× to 1.6×. ✅ 12–40× claim
  correctly withdrawn.
- **S (n=10 extension, 900 rows):** DP wins stay significant at both n;
  1 significance flip (credit/if/α=0.1). ✅ Corrected from earlier "6 flips".

## 3. PDF-rendering confirmation

- `make paper` ✅ (393 KiB, no errors)
- `make report` ✅ (320 KiB, no errors)
- Old "12–40×" absent from both PDFs ✅
- N1 uncorroborated ρ=0.668 claim removed/qualified; directional pattern
  (12/15 cells, ρ=0.131, p=0.8047) now what is reported ✅
- **Reproducibility note corrected in both PDFs**: the false "reproduces
  accuracy exactly and shifts DP by O(10⁻⁷), no DP conclusion depends on the
  fix" sentence is replaced with the actual TASK F deltas and the two flips ✅

## 4. What is fixed in this pass

1. `paper/sections/discussion.tex` — reproducibility note rewritten with real
   deltas + the adult/dp/α=0.1 and lsac/combined/α=0.1 flips.
2. `report/report.tex` — matching disclosure added to Other Limitations.
3. `results/task_f_verification_summary.md` — created (was missing).
4. `docs/TASKS_AL_VALIDATION.md` — corrected the overstrong "DP-attack
   headline is provably unaffected" claim.
5. Both PDFs rebuilt; `pytest tests/ -q` = 101 passed.

## 5. Verdict

**Conditionally ready to submit.** The science and builds are solid, and the
two overstated claims caught this week (12–40×, N1 ρ=0.668) are corrected.
Two decisions remain for Prof. Manisha before final submission:

1. **adult/dp/α=0.1 and lsac/combined/α=0.1 are marginal.** Under the locked
   grid they are significant (5/6 and 6/6); under the corrected-metric re-run
   they are not (4/6 and 4/6). The paper currently reports the locked grid and
   flags both cells honestly. Decide whether to (a) keep locked data with the
   marginality disclosed (current state) or (b) adopt the re-run file — do not
   submit without making this call.
2. **Whether to state Adult/DP α=0.1 as "5/6" (locked) with the re-run caveat**
   (current) or drop it to the honest "borderline / 4/6 under corrected code"
   framing. Current paper text already covers both; choose one headline phrasing.

The next agent/human should read both PDFs cover to cover once, confirm these
two cells read as marginal not as clean wins, and then submit. Do not touch
`results/canonical_tau1.json` or `results/utkface_canonical.json`.
