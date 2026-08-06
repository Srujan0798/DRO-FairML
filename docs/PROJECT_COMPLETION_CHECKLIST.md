# Everything left to complete the submission (2026-08-07)

Deadline: **Aug 10, 2026**. This is the full remaining work, in priority
order, so it can be split across independent agent CLIs with no overlap.
Ground rules for all of it are in `docs/TASKS_AL_VALIDATION.md` (pre-register,
report negatives, degeneracy guard, shared ablation lock, `pytest tests/ -q`
green before any commit).

**All compute described below is already scoped and pre-registered.** Nobody
needs to invent a new experiment — every task has a grid, a criterion, and a
deliverable already written down.

---

## Track 1 — AL validation and paper integration (the main event)

Ready-to-paste briefs: **`docs/AGENT_BRIEFS.md`**. Full spec: `docs/TASKS_AL_VALIDATION.md`.

| Task | Status | Who | Depends on |
|---|---|---|---|
| A — does AL generalise? | ✅ DONE (42/42) | — | — |
| C — μ sensitivity | ✅ DONE (90/90) | — | — |
| **C2 — AL × radius compound** | **open** | Agent 1 | none, can start now |
| **B — mechanism (why accuracy rises)** | **open** | Agent 2 | none, can start now |
| **D — paper/report integration** | **open** | Agent 2, after B | A, C (done) — unblocked |
| **E — independent adversarial review** | **open** | Agent 3 | none, can start now |

**A and C's headline result, already locked in:** DRO-FAIR-AL at **μ=20**,
scoped to **Adult, α≤0.2**, gives DP reduction of 70.8–81.7% over canonical
DRO with accuracy held or improved (6/6 seeds, p=0.0156). α=0.4 is unsafe (no
μ works); Credit is not rescued. This is the number and scope TASK D writes
into the paper — not the original μ=5.

**This track is the one thing that must land before submission** — it's the
answer to "why haven't you proposed an improvement," and Prof. Manisha has
already asked for it once.

---

## Track 2 — Prof. Manisha's two open decisions

Not agent work — needs her call, but each has a concrete recommendation
attached so it's a fast decision, not open-ended.

1. **Does AL go in the submission, and at what scope?**
   Recommendation (in `docs/MEMO_FOR_ADVISOR.md` §6a): yes, μ=20, Adult α≤0.2
   only, with α=0.4 and Credit stated as explicit exclusions.
2. **TASK F — canonical reproducibility gap.**
   `results/canonical_tau1.json` predates a k-NN metric fix; IF numbers on
   DP/COMBINED rows are floating-point noise (already disclosed, bogus
   significance stars already removed). Recommendation: re-run into a new
   file overnight (~6h, 540 rows) — DP results provably don't move. Can start
   the compute **before** she decides, since it writes to a new file and
   touches nothing locked.

**Action:** send her `docs/MEMO_FOR_ADVISOR.md` — it's written for exactly
this handoff, numbers pre-verified against `results/`.

---

## Track 3 — housekeeping that's easy to forget under deadline pressure

- **Agent S's n=10 seed extension** is mid-flight and *uneven* — seeds 6–9
  exist for some Adult cells (dp, if) but not Credit/LSAC, and not for
  `combined`. It's append-only and safely excluded from the paper by
  `loaders.load_canonical_tau1()`'s default (guarded by
  `tests/test_wilcoxon_seed_pairing.py::test_canonical_loader_excludes_n10_extension_by_default`).
  **Don't merge it into any paper table until it's complete and uniform
  across all three datasets** — a partial extension is worse than none.
- **Final full rebuild before submission:** `make paper && make report`, then
  open both PDFs and read them cover to cover once at the very end — not just
  the diffed section. Table/figure numbers can silently drift when multiple
  agents touch overlapping `.tex` sections in parallel; a last human/agent
  read-through is the actual gate, not just "did it compile."
- **Final full test suite** (`pytest tests/ -q`) after all tracks land, not
  just after each individual task — cross-task interaction (e.g. two agents
  editing `results.tex` in adjacent sections) is exactly what a merged final
  run catches and a per-task run doesn't.
- **`git log --oneline -20`** skim before submission to make sure nothing
  landed with a placeholder/TODO commit message that never got finished.

---

## What is explicitly NOT open work

Everything else is done and stable — do not re-run:
canonical grid (648 rows, locked seeds 0–5), kNN/tau/K-inner/empirical-radii/
random-vs-adversarial/attack-strength/radius-sensitivity ablations, UTKFace
canonical + multigroup + pixel-PGD, LSAC π-shrinkage and radii-fix
diagnostics, fairness-aggressiveness grid. All committed, all summarized, all
green under test. Re-running any of these without a specific new question is
wasted compute against the deadline.
