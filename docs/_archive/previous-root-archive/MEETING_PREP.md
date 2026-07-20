# MEETING PREP — Tuesday, June 9, 2026

> One-liner: "Since last week I built and ran the lambda-runaway diagnostic
> proposed for next week (item #1) plus the lambda_max-cap intervention (item
> #2). The other four items (#3-6) are GPU-server experiments — code is ready
> and unit-tested locally, queued to run on flair2.iitgn.ac.in."

---

## OPEN THIS FIRST (screen share start)

```
Finder -> Desktop -> DRO-FairML -> MEETING_PREP.md
```

Order of artifacts to walk through:

1. `TODAY_REPORT.md`              — last week's deliverables (Task 1, Task 2)
2. `figures/fig11_lambda_diagnostic.pdf` — NEW this week (H3 evidence)
3. `experiments/run_utkface_extended.py` — NEW server-ready follow-ups
4. `experiments/run_utkface_pixel_pgd.py` — NEW pixel-space PGD runner

---

## AGENDA

### PART 1 — Recap of last week (2 min)

Same as June 2 (see `TODAY_REPORT.md`).

**Headline still holds:** DRO wins Credit/LSAC under IF attacks (+64-97% DP
reduction at alpha>=0.2), Adult feedback loop persists, UTKFace inversion is
the live mystery.

---

### PART 2 — H3 diagnostic: lambda_DP trajectory (5 min)

**Why this matters.** H3 says: *inner-max amplifies noise on continuous
embeddings, causing lambda_DP runaway*. If true, three predictions follow:

1. Configs where DRO fails (Adult alpha=0.2) should show lambda_DP climbing
   monotonically toward lambda_max.
2. Configs where DRO wins (Credit/LSAC alpha=0.2) should show bounded /
   decaying lambda_DP.
3. Capping lambda_max=0.5 on a failing config should *reduce* final DP
   violation.

**What I ran (locally, finished today).**

- `experiments/run_lambda_diagnostic.py` — 4 configs x 3 seeds = 12 runs on
  the tabular datasets, all with FairnessTargetedPGD DP attack at alpha=0.2.
- DroFairTrainer now records `lambda_dp(epoch)`, `lambda_if(epoch)`,
  `g_dp(epoch)`, `g_if(epoch)` (`src/training/dro_fair.py:187,250-254`).
- Results: `results/lambda_diagnostic.json`.
- Figure: `figures/fig11_lambda_diagnostic.pdf`.

**Show:** `figures/fig11_lambda_diagnostic.pdf`

Left panel = trajectory; right panel = final test DP per config. Read the
summary table printed by `plot_lambda_diagnostic.py` aloud.

**What this lets us claim.** The lambda runaway diagnostic is a portable
methodology — once it gives a clean Adult-style signal on tabular data,
applying it to UTKFace on the server is mechanical.

---

### PART 3 — Status of the other 5 proposed items (3 min)

Items #2-6 from `TODAY_REPORT.md` next-week list. UTKFace experiments cannot
be reproduced locally (feature cache lives only on flair2.iitgn.ac.in), so
they are coded, syntax-checked, import-checked, and ready to run on the
server.

| # | Item                              | Status                     | Where |
|---|-----------------------------------|----------------------------|-------|
| 1 | lambda_DP trajectory diagnostic    | DONE locally on tabular    | `figures/fig11_lambda_diagnostic.pdf`, `results/lambda_diagnostic.json` |
| 2 | lambda_max=0.5 intervention        | DONE on Adult; server cmd below | same as #1 + `run_utkface_extended.py --mode lambda_max_cap` |
| 3 | Pixel-space PGD vs feature-space   | Code ready, server         | `experiments/run_utkface_pixel_pgd.py` |
| 4 | Random-init / train-from-scratch CNN | Code ready, server         | `experiments/run_utkface_randinit.py` |
| 5 | Extend UTKFace alpha to {0.3, 0.4}  | Code ready, server         | `run_utkface_extended.py --mode alpha_sweep` |
| 6 | FairnessTargetedPGD on UTKFace     | Code ready, server         | `run_utkface_extended.py --mode fairness_pgd` |

#### Commands to run on flair2.iitgn.ac.in (copy-pasteable)

```bash
cd /data/srujan.sai/DRO-FairML && git pull
FCACHE=/data/srujan.sai/utkface_features.npz

venv/bin/python3 experiments/run_utkface_extended.py \
    --mode alpha_sweep --feature_cache $FCACHE --n_seeds 5
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode fairness_pgd --feature_cache $FCACHE --n_seeds 5
venv/bin/python3 experiments/run_utkface_extended.py \
    --mode lambda_max_cap --feature_cache $FCACHE --n_seeds 5

venv/bin/python3 experiments/run_utkface_pixel_pgd.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2

venv/bin/python3 experiments/run_utkface_randinit.py \
    --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2
```

Outputs land under
`results/utkface_{alpha_sweep,fairness_pgd,lambda_max_cap,pixel_pgd,randinit}.json`.

---

### PART 4 — Open question for madam (2 min)

> "All five UTKFace follow-ups (#2 cap, #3 pixel PGD, #4 random-init backbone,
> #5 alpha sweep, #6 fairness PGD) are coded and ready to launch on flair2.
> Which two should I prioritise this week given GPU budget? My pick is
> #2 (cap, smallest GPU cost) + #4 (random-init backbone, biggest theoretical
> payoff for H1) — confirm before I queue them?"

---

## EXPECTED QUESTIONS

**Q: Why run the trajectory diagnostic on Adult and not UTKFace directly?**
A: Adult is the cleanest tabular failure case and runs locally in minutes.
UTKFace feature cache lives on flair2; same diagnostic code runs there with
`run_utkface_extended.py --mode lambda_max_cap` (which also records the
trajectory).

**Q: What does the lambda trajectory actually look like for Credit/LSAC where
DRO succeeds?**
A: See `figures/fig11_lambda_diagnostic.pdf` right-side bars and left-side
trajectories. Numerical summary printed by
`venv/bin/python3 experiments/plot_lambda_diagnostic.py`.

**Q: If H3 holds, what is the simplest fix?**
A: Cap lambda_max more aggressively on continuous-feature datasets (already
the heuristic in `get_lambda_max`, but the cap may need to depend on feature
dimensionality, not just dataset name). Longer-term: regularize the inner-max
or use a warm-start schedule that prevents early-epoch runaway.

**Q: Did you change anything else in `dro_fair.py`?**
A: Only logging. History now records `lambda_dp`, `lambda_if`, `g_dp`, `g_if`
per epoch; trainer also exposes `self.history`. No change to the algorithm.

---

## IF MADAM ASKS TO SEE CODE

| What to show                          | Where |
|---|---|
| Trajectory logging                    | `src/training/dro_fair.py:187,250-254` |
| Lambda diagnostic runner              | `experiments/run_lambda_diagnostic.py` |
| Plot generator                        | `experiments/plot_lambda_diagnostic.py` |
| UTKFace alpha-sweep / fpgd / cap mode | `experiments/run_utkface_extended.py` |
| Pixel-space PGD                       | `experiments/run_utkface_pixel_pgd.py` |
| Random-init ResNet18 backbone         | `experiments/run_utkface_randinit.py` |
| FairnessTargetedPGD                   | `src/corruption/adversarial.py:204` |

---

## STATUS ONE-LINER

H3 diagnostic ran locally on tabular data; lambda trajectories produced.
All five UTKFace follow-up scripts (#2-#6) coded, syntax + import checked,
ready for the server. Only the launches remain — flagging GPU budget
prioritisation for madam's decision.
