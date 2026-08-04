# DRO-FairML — Session Handoff

**Date written:** 2026-06-08
**Author of this doc:** Claude (Opus 4.7), continuing a chain of sessions on this repo.
**Audience:** the next agent (you) picking up the work, or Srujan when he hands it over.

> Read this top-to-bottom before touching anything. Section order is:
> 1. Repo state (branch, commits, untracked)
> 2. Project context (what DRO-FairML is, what madam asked, the live open question)
> 3. What this session did (commit `7ff54b9`)
> 4. Untracked files left in the working tree (DO NOT auto-commit — review)
> 5. The 6-item next-week list + which are done / in progress / not started
> 6. Exact server commands ready to run
> 7. Pitfalls, gotchas, and DO-NOT-DO rules from memory

---

## 1. Repo state at handoff

- **Repo:** `/Users/srujansai/Desktop/DRO-FairML`
- **Remote:** `https://github.com/Srujan0798/DRO-FairML.git`
- **Branch:** `main`, in sync with `origin/main` until the new commit below — that commit is **local only**, not pushed.
- **Latest commit (this session):** `7ff54b9` — "Add lambda trajectory logging + UTKFace attack/lambda_max flags"
- **Previous commit:** `fc43e6f` — α=0.1 robustness tax clarification (Jun 2)
- **Working tree:** clean for the 2 files I modified. 6 untracked files (see §4).

### Pushing
The new commit `7ff54b9` has **not been pushed**. Decide whether to push before/after also handling the untracked files. Server pulls from `origin/main`, so until you push, the new flags won't be available on `flair2.iitgn.ac.in`.

```bash
# When ready:
git push origin main
```

---

## 2. Project context (full briefing — do not skim)

### What DRO-FairML is
Implementation + empirical evaluation of **DRO-FAIR (Algorithm 1)** from an ICML submission (PDF lives at `ICML_submission.pdf`, original v1.0 report at `submission/report.pdf`).

**Srujan's contribution** (per professor's assignment): replace the paper's random-noise corruption model with **adversarial corruption** — PGD on features, coordinated label flips, minority-targeted attribute flips. Evaluated on Adult, Credit, LSAC.

### Last meeting (June 2, 2026)
Madam (advisor) asked two follow-ups:

1. **PGD attacks targeting fairness metrics** — three modes (DP-only, IF-only, combined), evaluate Naive vs DRO on Adult/Credit/LSAC.
2. **UTKFace on the GPU server** — same DRO-FAIR pipeline but on face images (ResNet18 features → 512-d → Naive/DRO heads). Argument: previous study was small-tabular only; go bigger.

### What's already shipped (before this session)
- ✅ `src/corruption/adversarial.py::FairnessTargetedPGD` — three attack modes
- ✅ 270 tabular runs (3 datasets × 3 attacks × 2 methods × 3 α × 5 seeds) → `results/fairness_pgd_results.json`, `results/fairness_pgd_wilcoxon.csv`, `results/fairness_pgd_summary.csv`
- ✅ 15 UTKFace baseline runs (5 seeds × 3 α × 2 methods) on GPU → `results/utkface_results.json`
- ✅ Figures: `figures/fig8_attack_defense_matrix.pdf`, `figures/fig9_fairness_pgd_curves.pdf`, `figures/fig10_utkface_curves.pdf`
- ✅ Two reports written: `TODAY_REPORT.md` (weekly progress, paper-style), `MEETING_PREP.md` (untracked screen-share script for the Jun 2 meeting)

### The live open question (what madam is waiting on)
**DRO behavior INVERTS between tabular and image features.**

| Setting | DRO vs Naive |
|---|---|
| Credit α=0.3 / IF attack | DRO **−97.5%** DP (p=0.031) — wins |
| LSAC α=0.3 / IF attack | DRO **−96.2%** DP — wins |
| Adult any α / DP attack | DRO loses — λ_DP runaway / feedback loop |
| UTKFace α=0.1 | DRO **+22% worse** DP — inversion |
| UTKFace α=0.2 | DRO **+15% worse** DP — inversion |

Three hypotheses on the table (from `TODAY_REPORT.md`):
- **H1** — ResNet18 features are gender-agnostic (ImageNet-pretrained → DRO has no demographic axis to anchor on).
- **H2** — Feature-space attacks (cached 512-d) don't simulate real corruption; real attacker would perturb raw pixels.
- **H3** — Inner maximization amplifies noise on continuous embeddings; λ_DP over-grows.

None of the three has been tested yet. **This is the next-week scope.**

### α=0.1 "robustness tax" answer (madam already asked)
DRO has near-zero advantage at α=0.1 across all datasets. The answer Srujan committed (`fc43e6f`): TV radii ρ_DP, ρ_IF depend only on α (not attack design), so DRO's defensive envelope is fixed by α. At α=0.1, naive doesn't fail enough for DRO's worst-case-prep to pay off — textbook DRO robustness tax. Important: do **not** re-explain this as "the attack is too weak."

---

## 3. What this session did → commit `7ff54b9`

### Files changed (2)

#### `src/training/dro_fair.py`
Added per-epoch logging to the `history` dict returned by `DroFairTrainer.fit()`:

```python
history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': [],
           'lambda_dp': [], 'lambda_if': [], 'g_dp': [], 'g_if': []}
# ...
history['lambda_dp'].append(float(lambda_dp.item()) if self.use_dp else 0.0)
history['lambda_if'].append(float(lambda_if.item()) if self.use_if else 0.0)
history['g_dp'].append(float(g_dp.item()) if self.use_dp else 0.0)
history['g_if'].append(float(g_if.item()) if self.use_if else 0.0)
```
- Backward-compatible: nothing existing reads or breaks on the new keys.
- Purpose: H3 diagnostic — does λ_DP run away on UTKFace like it does on Adult?

#### `experiments/run_utkface.py`
Three new CLI flags (no change to default behavior):

```bash
--attack {adversarial,dp,if,combined}    # default: adversarial
--lambda_max FLOAT                       # default: 1.5; test H3 with 0.5
--save_lambda_history                    # persists per-epoch λ_DP/λ_IF/g_DP/g_IF
--output PATH                            # override default output filename
```

Output filename auto-tags: `results/utkface_results_<attack>_lmax<N>.json` so parallel runs don't overwrite.

**`run_single_utkface_experiment(...)`** now takes:
- `lambda_max` (piped into `DroFairTrainer(lambda_max=...)`)
- `attack` (switches between `AdversarialCorruptor` and `FairnessTargetedPGD`)
- `save_lambda_history` (persists `dro_history` into `results['dro']['lambda_history']`)

### Smoke tests run (synthetic UTKFace fallback — real images not local)

```bash
venv/bin/python3 experiments/run_utkface.py --smoke --attack if --lambda_max 0.5 \
  --save_lambda_history --output /tmp/utk_smoke.json
# → 4s. 60 epochs of λ_dp logged, λ_dp reached 0.043 (well under 0.5 cap on synthetic).

venv/bin/python3 experiments/run_utkface.py --smoke --output /tmp/utk_smoke_default.json
# → 3s. Default path unchanged, backward-compat confirmed.

venv/bin/python3 -c "from experiments.run_fairness_pgd import run_single_experiment; ..."
# → tabular fairness-PGD path still runs (didn't break the existing pipeline).
```

**Why not run real experiments locally:** no UTKFace data on this laptop (only `docs/UTKFACE_PIPELINE.md`), no CUDA (only MPS). Real data + GPU = `flair2.iitgn.ac.in`. Server data path: `/data/srujan.sai/UTKFace/`, feature cache lives at `/data/srujan.sai/utkface_features.npz`.

---

## 4. Untracked files left in the working tree (REVIEW FIRST)

These appeared in `git status` during this session but I **did not write them in this session** — they look like work from an earlier (parallel? auto?) session that wasn't yet committed. Their docstrings are well-aligned with the 6 next-week items, but I haven't read every line.

| File | Lines | Purpose (per docstring) | Item |
|---|---|---|---|
| `MEETING_PREP.md` | 158 | Screen-share script for the Jun 2 meeting (already happened) | n/a |
| `experiments/run_lambda_diagnostic.py` | 121 | Records λ_DP trajectory across Adult/Credit/LSAC at α=0.2, with and without `lambda_max=0.5` cap | #1, #2 |
| `experiments/plot_lambda_diagnostic.py` | ~30 | Plots the trajectory + final-DP bar chart from above | #1 |
| `experiments/run_utkface_extended.py` | 225 | Three modes: `alpha_sweep` (α∈{0.3,0.4}), `fairness_pgd` (DP/IF/combined on images), `lambda_max_cap` (H3 test on UTKFace) | #2, #5, #6 |
| `experiments/run_utkface_pixel_pgd.py` | 279 | Pixel-space PGD through ResNet18 vs feature-space attack | #3 (H2) |
| `experiments/run_utkface_randinit.py` | ~?  | Random-init ResNet18 trained from scratch; tests whether DRO inversion goes away with non-ImageNet features | #4 (H1) |

### What to do with them
1. **Read each one.** They reference H1/H2/H3 and item numbers — but verify they actually do what their docstrings claim before relying on results.
2. **Sanity-smoke-test each** with synthetic / small data before pushing to server.
3. **Commit individually** with explicit messages — don't `git add .` blanket.
4. `MEETING_PREP.md` — the meeting already happened. Either commit as a record, move to `docs/archive/`, or delete. Don't leave it floating.

---

## 5. The 6-item next-week list (from TODAY_REPORT.md §"Proposed Next Week")

| # | Item | Status | Where the code is |
|---|---|---|---|
| 1 | λ_DP trajectory diagnostic on UTKFace | **Wired** (this session) + **runner exists** (untracked) | `dro_fair.py` logs it; `run_utkface.py --save_lambda_history` exposes it; `run_lambda_diagnostic.py` (untracked) does tabular comparison |
| 2 | UTKFace with `λ_max = 0.5` (test H3) | **Wired** (this session) | `run_utkface.py --lambda_max 0.5`; also a mode in `run_utkface_extended.py` (untracked) |
| 3 | Pixel-space PGD vs feature-space (test H2) | **Runner exists** (untracked) | `run_utkface_pixel_pgd.py`; also a new module `src/corruption/image_pgd.py` already exists in repo |
| 4 | Random-init backbone CNN (test H1) | **Runner exists** (untracked) | `run_utkface_randinit.py` |
| 5 | UTKFace α∈{0.3, 0.4} | **Wired** (this session via `--alphas`) + dedicated mode in untracked runner | `run_utkface.py --alphas 0.3 0.4`; `run_utkface_extended.py --mode alpha_sweep` |
| 6 | FairnessTargetedPGD on UTKFace | **Wired** (this session) | `run_utkface.py --attack {dp,if,combined}`; also `run_utkface_extended.py --mode fairness_pgd` |

**Reality:** all 6 are either wired in `run_utkface.py` or have a dedicated untracked runner. Nothing is finished — nothing has been run on real UTKFace data on the GPU server since this session started.

---

## 6. Exact server commands ready to run

**SSH target:** `flair2.iitgn.ac.in`
**Project on server:** `/data/srujan.sai/DRO-FairML` (presumably; verify)
**Feature cache:** `/data/srujan.sai/utkface_features.npz` (referenced by untracked runners)
**Raw images:** `/data/srujan.sai/UTKFace/`

After pushing `7ff54b9`:

```bash
# On server:
cd /data/srujan.sai/DRO-FairML && git pull

# --- Items doable with the committed run_utkface.py changes ---

# Item 1: λ trajectory on UTKFace (default adversarial attack)
venv/bin/python3 experiments/run_utkface.py \
  --alphas 0.1 0.2 --n_seeds 5 --save_lambda_history \
  --output results/utkface_lambda_traj.json

# Item 2: H3 quick test — cap λ_max at 0.5
venv/bin/python3 experiments/run_utkface.py \
  --alphas 0.1 0.2 --n_seeds 5 --lambda_max 0.5 --save_lambda_history \
  --output results/utkface_lmax05.json

# Item 5: extend α sweep to {0.3, 0.4}
venv/bin/python3 experiments/run_utkface.py \
  --alphas 0.3 0.4 --n_seeds 5 \
  --output results/utkface_alpha_high.json

# Item 6: FairnessTargetedPGD on UTKFace, all three modes
for mode in dp if combined; do
  venv/bin/python3 experiments/run_utkface.py \
    --alphas 0.1 0.2 0.3 --n_seeds 5 --attack $mode \
    --output results/utkface_fpgd_$mode.json
done
```

For items 3 and 4, **first read** the untracked runners (`run_utkface_pixel_pgd.py`, `run_utkface_randinit.py`) and verify they don't crash on a smoke run before committing to a full sweep.

---

## 7. Pitfalls, gotchas, DO-NOT-DO rules

### From persistent memory (these are real corrections from past sessions)

- **NEVER suggest random corruption.** Adversarial corruption is the whole point of the project. There is a `RandomCorruptor` class in `src/corruption/adversarial.py` — it exists only as a comparison baseline. Do not propose reverting to it.
- **Lambda NOT in inner gradient.** In `dro_fair.py` Step 4 (inner max on p), the loss being differentiated is the *unweighted* DP/IF violation, not λ*violation. This is deliberate — λ doesn't change the argmax, and including it caused instability in past experiments. Leave it as is.
- **Codebase finalized 2026-05-14.** `epochs=60` and `K_inner=10` are mandatory for tabular runs. Do not lower them silently to make smoke tests faster — use `--smoke` paths that explicitly mark themselves as smoke.
- **No publicity.** This repo is for the professor only. **Do not** write blog posts, profile READMEs, Pages sites, or LinkedIn content. (The handoff doc you're reading is fine — it's internal.)

### Engineering pitfalls

- **`get_temperature(alpha)`** returns 1.0 if α≥0.4 else 100.0. The α=0.4 results in the new sweep will use τ=1, not τ=100 — different regime. Don't blame data if α=0.4 results look strange.
- **`get_lambda_max(dataset, alpha)`** in tabular runners returns 0.5 for Adult α≥0.2 (because of the λ runaway). UTKFace has no such override yet — `run_utkface.py` uses a single `--lambda_max` value across all configs.
- **JSON serialization** in `run_utkface.py` coerces only `clean`/`corrupted` sub-dicts to floats. The new `lambda_history` sub-dict relies on my logging code already storing Python floats. If you add new fields, either explicitly cast or extend the coercion loop at the end of `run_single_utkface_experiment`.
- **The synthetic UTKFace fallback** in `run_utkface.py::_make_synthetic_utkface` uses random Gaussians + random labels — **smoke results are meaningless** for any scientific claim. Use them only to confirm code doesn't crash.
- **`get_dataset('utkface', ...)`** raises `RuntimeError('No UTKFace images found in data/raw')` when images are absent. The runner catches this and falls back to synthetic. On the server, the path must resolve — confirm the dataset loader points to the right directory.

### Decision points for the next agent

- **Push or not?** `7ff54b9` is local only. Push when you're confident in it.
- **Untracked runners** — read, verify, smoke, then commit. Don't trust docstrings.
- **`MEETING_PREP.md`** — decide: commit as-is, archive, or delete.
- **The "DRO inverts on images" story** — the professor (madam) is waiting on the answer. The lowest-cost path to an answer is item #2 (λ_max=0.5 sweep): one server run, ~hours, directly tests H3.

---

## 8. File map (read these to orient quickly)

| File | What it is | Read if you're working on |
|---|---|---|
| `TODAY_REPORT.md` | Most recent weekly report (paper-style; written for the meeting) | Anything — start here |
| `MEETING_PREP.md` (untracked) | Screen-share script from Jun 2 | Understanding what was said in the meeting |
| `src/training/dro_fair.py` | Core Algorithm 1 implementation | Anything DRO-related |
| `src/training/naive_fair.py` | Naive baseline (same training loop, no inner max) | Comparisons |
| `src/corruption/adversarial.py` | `AdversarialCorruptor` + `FairnessTargetedPGD` + `RandomCorruptor` | Corruption / attack work |
| `src/corruption/image_pgd.py` | Image-space PGD (referenced by untracked pixel-PGD runner) | Item #3 |
| `src/data/datasets.py` | Dataset loaders incl. UTKFace | Data path debugging |
| `experiments/run_utkface.py` | The UTKFace runner (now with new flags) | Items #1, #2, #5, #6 |
| `experiments/run_fairness_pgd.py` | The tabular Fairness-PGD runner (already used to produce results) | Replicating the 270-run table |
| `results/utkface_results.json` | The 15 baseline UTKFace results from before this session | Comparisons |
| `results/fairness_pgd_results.json` | The 270 tabular runs | Comparisons / paper |
| `results/fairness_pgd_wilcoxon.csv` | Significance tests | Headline claims |
| `figures/fig8_*`, `fig9_*`, `fig10_*` | Existing figures referenced in the report | Citing existing visuals |
| `ICML_submission.pdf` | The paper this work is based on | Algorithm details |
| `submission/report.pdf` | The original v1.0 student report | History |
| `docs/UTKFACE_PIPELINE.md` | UTKFace dataset setup notes | Data path / preprocessing |
| `docs/UTKFACE_RESULTS.md` | UTKFace results notes | Background |

---

## 9. TL;DR for the next agent

1. **Don't push `7ff54b9` blindly** — read the diff first (`git show 7ff54b9`).
2. **Read each untracked file in `experiments/`** before committing or running. Their docstrings look right but I haven't audited the bodies.
3. **The cheapest scientific next step is item #2** (λ_max=0.5 sweep on UTKFace) — one CLI flag, one server run, directly tests H3.
4. **All real experiments need flair2.iitgn.ac.in.** Local laptop can only validate code shape against synthetic data.
5. **Never revert to random corruption. Never write public content about this project.**

Good luck.
