# DRO-FairML — Session Handoff #2

**Date:** 2026-06-08 (evening, ~16:30 IST)
**Author:** Claude (Opus 4.7), session #2 — continues from `HANDOFF.md`
**Audience:** the next agent (or Srujan when re-opening). **Meeting with madam is tomorrow, June 9.**

> Read `HANDOFF.md` first (it has the project bible). This doc only covers
> what changed in **this** session and what's left to land. Do NOT re-derive
> the project context — it's in HANDOFF.md §1–§2.

---

## TL;DR — 90 seconds

- Five new server-ready experiment scripts written, syntax + import + history-smoke checked, **NOT committed**.
- One new diagnostic plot script written, **NOT committed**.
- One unit-test-style verifier added so the server can sanity-check before queuing 6+ hours of GPU.
- `MEETING_PREP.md` fully rewritten for tomorrow (June 9) — replaces the June 2 version.
- The local lambda diagnostic was **started but never completed** (process hung at ~1% CPU after 19 min, killed). See §4 — do not interpret this as a "result." The diagnostic must be re-run on the server, or locally with shorter epochs.
- Nothing pushed. Nothing committed. Working tree is dirty with 7 untracked files and HANDOFF.md (also untracked from prior session).

---

## 1. Repo state at handoff

- **Branch:** `main`. Local HEAD = `7ff54b9` (lambda trajectory logging — committed by **prior** session, see HANDOFF.md §3). HEAD is **ahead of `origin/main`** by 1 commit (HANDOFF.md says it never pushed; verify with `git log origin/main..HEAD`).
- **Working tree (added in this session unless noted):**

```
?? HANDOFF.md                                  (from prior session — never tracked)
?? HANDOFF_2.md                                (this doc)
?? MEETING_PREP.md                             (REWRITTEN today for June 9)
?? experiments/run_lambda_diagnostic.py        (NEW — H3 trajectory diagnostic, tabular)
?? experiments/plot_lambda_diagnostic.py       (NEW — figure generator)
?? experiments/run_utkface_extended.py         (NEW — 3 modes: alpha_sweep, fairness_pgd, lambda_max_cap)
?? experiments/run_utkface_pixel_pgd.py        (NEW — pixel-space PGD via ResNet18)
?? experiments/run_utkface_randinit.py         (NEW — random-init ResNet18 backbone)
?? experiments/verify_server_scripts.py        (NEW — pre-flight check for server)
```

- No tracked file was modified in this session. `git diff HEAD` is empty.
- `MEETING_PREP.md` from prior session (June 2) was **overwritten** in place — the June 2 version is gone unless recovered from git history.

---

## 2. What the meeting needs (June 9, 10:00 IST — confirm time)

Madam will ask: *"What did you do this week on the three hypotheses?"*

**The answer Srujan should give** (already drafted in `MEETING_PREP.md`):

1. **H3 diagnostic methodology built** — `DroFairTrainer` now records `lambda_dp`, `lambda_if`, `g_dp`, `g_if` per epoch (committed in `7ff54b9`). A 4-config trajectory experiment exists (`run_lambda_diagnostic.py`).
2. **All five follow-up experiments coded** — see §3 below. Each script's `--help` works (verified). UTKFace scripts are server-only (no local data / no CUDA).
3. **No new empirical results yet** — the local diagnostic hung; the server runs have not been launched.
4. **Decision needed:** GPU budget prioritisation across the five items.

**Risk:** if madam asks "what does the lambda trajectory actually look like on Adult?" Srujan cannot show a figure. The current honest answer: *"The diagnostic ran on three CPU configs for ~20 min then hung — I'll re-launch with shorter epochs after the meeting and have results by Thursday."* Do **not** pretend results exist.

---

## 3. The five new experiment scripts (all server-ready, all unrun)

All scripts live in `experiments/`. All are stand-alone (`python3 experiments/<file>.py --help` works after a 30–90s import). Outputs land in `results/`.

### 3.1 `run_lambda_diagnostic.py` (item #1, **tabular**, local-runnable)
**Tests H3 on tabular data.** 4 configs × 3 seeds × 60 epochs:

| tag                 | dataset | alpha | lambda_max | known behavior |
|---|---|---|---|---|
| `adult_lmax1.5`     | adult   | 0.2   | 1.5        | DRO fails (feedback loop) |
| `adult_lmax0.5`     | adult   | 0.2   | 0.5        | intervention test |
| `credit_lmax1.5`    | credit  | 0.2   | 1.5        | DRO wins |
| `lsac_lmax1.5`      | lsac    | 0.2   | 1.5        | DRO wins |

Attack used: `FairnessTargetedPGD(target_metric='dp')`. Output: `results/lambda_diagnostic.json`.

**Known issue:** ran for 19 min on Python 3.14 + CPU and hung at ~1% CPU after the first config header was printed. Suspected cause: `sklearn.NearestNeighbors` + `project_simplex_l1_ball(max_iter=500)` inside `K_inner=10` × 60 epochs on Adult (~30k samples). Either:
- Lower `K_inner` to 3 and `epochs` to 20 for a fast local sanity pass
- Or run on the server where it should be ≤10 min total
- Or rewrite the simplex projection to use a vectorised version

### 3.2 `run_utkface_extended.py` (items #2, #5, #6, **server-only**)
Three modes, each accepts `--feature_cache /data/srujan.sai/utkface_features.npz --n_seeds 5`:

```bash
--mode alpha_sweep      # extends UTKFace to alpha ∈ {0.3, 0.4}                 (item #5)
--mode fairness_pgd     # runs DP / IF / combined PGD attacks on UTKFace          (item #6)
--mode lambda_max_cap   # compares lambda_max=1.5 vs 0.5 at alpha ∈ {0.1, 0.2}    (item #2 / H3)
```

Outputs: `results/utkface_alpha_sweep.json`, `results/utkface_fairness_pgd.json`, `results/utkface_lambda_max_cap.json`. The `lambda_max_cap` mode also records `dro_history.lambda_dp` per epoch, so the same plot script works.

### 3.3 `run_utkface_pixel_pgd.py` (item #3, **server-only**)
Tests H2. End-to-end pipeline:
1. Train a classifier head on clean ResNet18 features.
2. Pick top-α n train images by classification margin.
3. Run PGD in **pixel space** through `ResNet18 ∘ head` (`PixelToLogit` wrapper).
4. Re-extract features from attacked pixels.
5. Train Naive + DRO heads on attacked features.

Defaults: `eps=4/255, steps=10, batch=64`. Output: `results/utkface_pixel_pgd.json`. Bottleneck = pixel PGD on 24k images, ~25 min/alpha/seed on L40S.

### 3.4 `run_utkface_randinit.py` (item #4, **server-only**)
Tests H1. Trains `ResNet18(weights=None)` end-to-end on UTKFace gender for `--backbone_epochs 15` (cosine LR), then uses the trained backbone as feature extractor, then runs Naive + DRO heads on attacked features. Output: `results/utkface_randinit.json`. Bottleneck = backbone training, ~30 min/seed on L40S.

### 3.5 `verify_server_scripts.py` (housekeeping)
Run this on the server immediately after `git pull` and before queuing the long jobs. Checks parse, `--help` works (180s timeout), and runs a 3-epoch `DroFairTrainer` smoke test to confirm the new history keys (`lambda_dp`, `lambda_if`, `g_dp`, `g_if`) appear. Exits non-zero on any failure.

---

## 4. The diagnostic that didn't complete

| Attempt | Background ID | Outcome |
|---|---|---|
| 1 | `b2jeuffqc` | ran 8 min at ~90% CPU, **exited 0** with **0 bytes of output**. Cause unclear — `tee` may have swallowed everything, or python crashed silently after the import phase. |
| 2 | `b7dt64cck` | killed early via `head -5` (exit 144 = SIGPIPE). 0 bytes. |
| 3 | `bbcq6537t` | 19 min, hung at ~1% CPU after printing `[adult_lmax1.5] dataset=adult alpha=0.2 lambda_max=1.5`. `sample` confirmed it was in numpy/scipy land — likely the simplex projection inside `K_inner`. Killed manually. |

**No `results/lambda_diagnostic.json` exists. No `figures/fig11_lambda_diagnostic.pdf` exists.** Do not claim otherwise to madam.

**Recommended next attempt:**
```bash
# Locally — short-epoch sanity pass
venv/bin/python3 experiments/run_lambda_diagnostic.py
# Edit run_lambda_diagnostic.py: epochs=20, K_inner=3, seeds=[0, 1] before running.
# Then once the figure looks right, re-run with full epochs on the server.
```

Or on flair2 once GPU access is restored — the GPU will absorb the simplex projection cost without issue.

---

## 5. Decisions handed back to Srujan / next agent

1. **Push or not?** Working tree is dirty with 7 files. Suggested commit message:

   ```
   Add 5 follow-up experiment scripts + lambda diagnostic + meeting prep

   - run_lambda_diagnostic.py / plot_lambda_diagnostic.py: H3 trajectory on
     Adult/Credit/LSAC with lambda_max ∈ {1.5, 0.5}, alpha=0.2 DP attack
   - run_utkface_extended.py: 3 modes — alpha sweep to 0.3/0.4 (item #5),
     fairness PGD (item #6), lambda_max cap (item #2 / H3)
   - run_utkface_pixel_pgd.py: pixel-space PGD via ResNet18 (item #3 / H2)
   - run_utkface_randinit.py: random-init ResNet18 backbone (item #4 / H1)
   - verify_server_scripts.py: pre-flight check for the server
   - HANDOFF.md, HANDOFF_2.md, MEETING_PREP.md: meeting + agent handoff docs

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```

   Srujan answered **commit + push** to my earlier question, but the
   diagnostic never completed so the figure that was supposed to land in
   the same commit is missing. Decision for next agent: commit now without
   the figure, or wait until the diagnostic completes and includes the
   figure.

2. **GPU access.** Per Srujan today: GPU is **not available right now**.
   None of the 4 UTKFace scripts can run anywhere except flair2.

3. **The MEETING_PREP.md framing.** It is written assuming the lambda
   diagnostic figure exists. If the figure does NOT exist by meeting time,
   edit `MEETING_PREP.md` §2 to drop the figure reference and say "diagnostic
   methodology ready; results queued for server."

---

## 6. Exact commands ready (copy-pasteable)

### Locally (after fixing the hang)
```bash
cd /Users/srujansai/Desktop/DRO-FairML
# (edit run_lambda_diagnostic.py to epochs=20, K_inner=3 first)
venv/bin/python3 experiments/run_lambda_diagnostic.py
venv/bin/python3 experiments/plot_lambda_diagnostic.py
# → figures/fig11_lambda_diagnostic.{pdf,png}
```

### On flair2.iitgn.ac.in (when GPU returns)
```bash
cd /data/srujan.sai/DRO-FairML && git pull
venv/bin/python3 experiments/verify_server_scripts.py        # pre-flight
FCACHE=/data/srujan.sai/utkface_features.npz

# Item #2 / H3 — smallest cost, biggest diagnostic value
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode lambda_max_cap --feature_cache $FCACHE --n_seeds 5

# Items #5, #6 — extends existing results
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode alpha_sweep --feature_cache $FCACHE --n_seeds 5
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode fairness_pgd --feature_cache $FCACHE --n_seeds 5

# Items #3, #4 — expensive (pixel-level + scratch training)
venv/bin/python3 experiments/run_utkface_pixel_pgd.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2
venv/bin/python3 experiments/run_utkface_randinit.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2
```

---

## 7. DO-NOT-DO list (carried from memory + HANDOFF.md)

- DO NOT revert to `RandomCorruptor`. Adversarial corruption is the project's reason for existing. (Memory: `feedback_no_random_corruption.md`.)
- DO NOT multiply lambda into the inner-max gradient step. The current omission is deliberate. (Memory: `feedback_lambda_inner_gradient.md`.)
- DO NOT change `epochs=60` or `K_inner=10` as the default in `dro_fair.py` — those are the mandated training settings. (Memory: `project_codebase_state.md`.) You may pass smaller values via constructor args for *diagnostic* runs only.
- DO NOT publicise this repo. No blog post, no profile README pin, no LinkedIn, no Pages site. Audience = professor + madam only. (Memory: `feedback_no_publicity.md`.)
- DO NOT claim the lambda diagnostic produced results. It did not (see §4).

---

## 8. Open questions for the next agent

1. Why did Python 3.14 + macOS + the simplex projection stall? Worth a 10-min profiling pass before re-running.
2. Is the GPU server access still revoked, or returning soon? Drives the priority of fixing the local hang vs just queueing on flair2.
3. Should `MEETING_PREP.md` §2 be softened (no figure) before the meeting, or is the diagnostic going to be re-attempted locally tonight?

---

## 9. File index (quick map)

| Item | File | Status |
|---|---|---|
| HANDOFF (prior) | `HANDOFF.md` | untracked, **read first** |
| HANDOFF (this) | `HANDOFF_2.md` | this file |
| Meeting prep    | `MEETING_PREP.md` | untracked, rewritten for June 9 |
| H3 trajectory runner | `experiments/run_lambda_diagnostic.py` | untracked, hangs on Python 3.14 |
| H3 plot          | `experiments/plot_lambda_diagnostic.py` | untracked |
| UTKFace 3-mode runner | `experiments/run_utkface_extended.py` | untracked, server-only |
| UTKFace pixel PGD | `experiments/run_utkface_pixel_pgd.py` | untracked, server-only |
| UTKFace random-init | `experiments/run_utkface_randinit.py` | untracked, server-only |
| Server pre-flight | `experiments/verify_server_scripts.py` | untracked |
| DRO instrumentation | `src/training/dro_fair.py` | **committed in `7ff54b9`** |
| Last weekly report | `TODAY_REPORT.md` | committed |
| Memory store | `~/.claude/projects/-Users-srujansai-Desktop-DRO-FairML/memory/` | read before answering |

End of handoff.
