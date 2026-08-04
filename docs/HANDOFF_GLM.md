# HANDOFF — GLM: finish the CPU ablation queue

**Your lane: Mac CPU only.** You do not touch flair2, GPU code, or any file Grok's lane
owns (see docs/HANDOFF_GROK.md). You do not touch `paper/*.tex`, `report/*.tex`, or
`STATUS.md` — integration is a separate later pass so two agents never edit the same
`.tex` file at once (that collision already happened once tonight and broke the paper).

## Locked truth — read once, don't re-derive

- `results/canonical_tau1.json`: **540 rows locked** (τ=1, K=10, n=6, seeds 0-5). A
  parallel n=10 extension has appended seeds 6-9 for some cells (file may show >540) —
  **all claims still use seeds 0-5 / the original 540.** Never edit or shrink this file.
- `results/utkface_canonical.json`: 90/90 REAL rows, complete. Not your lane (Grok owns
  UTKFace/GPU work) — read-only reference if you need it.
- Central finding: fixed τ=1 makes DRO beat Naive on Adult/Credit DP+Combined at α≤0.2
  (Adult/DP α=0.1 is 5/6, not 6/6 — everywhere else 6/6). LSAC/DP is degenerate (no fix
  found — L2 tested and confirmed, see `results/lsac_radii_summary.md`). IF is MIXED.
  α≥0.3 falls below the constant predictor on Adult/Credit only (LSAC is pinned at
  baseline, not below).

## The one hard rule

**Never run an ablation script directly (`python3 experiments/run_a1_knn.py` etc.).**
Always go through the orchestrator or the shared lock — running scripts directly bypasses
nothing (the lock in `experiments/run_ablation_parallel.py` still queues you), but you
gain nothing either, and it's how tonight's confusion started. Check state first:

```bash
ps aux | grep -iE "orchestrate_wave1|run_a[0-9]|run_n[0-9]|run_l2|run_s_n10"
```

- If `orchestrate_wave1.sh` (or a `run_*.py` job) is already running → **let it run.**
  Don't relaunch, don't kill it. Check back with the same `ps` command or
  `tail -f logs/orchestrate_wave1.log`.
- If nothing is running → `bash scripts/orchestrate_wave1.sh` (it's resume-safe;
  finished jobs are skipped instantly, in-progress jobs resume exactly where they left
  off — checkpointed after every single result via `atomic_save`).

## What's already done — do not re-run

| File | Status |
|---|---|
| `results/if_wilcoxon_summary.txt`, N4 IF@α=0.3 analysis | DONE |
| `results/lsac_radii_fix.json` | DONE (120/60 — over target, that's fine, it's evidence the hypothesis was tested thoroughly) |
| `results/extended_datasets.json` (COMPAS + German) | DONE — German replicates the DRO pattern, COMPAS is ambiguous. Report both honestly. |
| `results/lambda_grid.json` | DONE 72/72 — no (λ,lr) beats default, no α=0.3 rescue |

## What's left — the orchestrator queue, in order

| Job | Target | Rows so far (check live, this is a snapshot) |
|---|---|---|
| N2-HighAlpha | 120 | ~26+ |
| A1-kNN | 360 | 48 |
| A2-Tau | 360 | 76 |
| A4-RvA | 144 | 43 |
| A5-Empirical | 180 | 69 |
| N5-Kinner | 180 | 24 |
| N1-AttackStrength (a) | 144 | 22 |
| N1-AttackStrength (b) | 180 | 22 |
| S-N10-Extension | 900 total | 560 |

Early finding worth knowing (from A4 partial data): the "12–40× stronger than random"
claim quoted to Kuldeep on Jun 16 does **not** hold under the canonical protocol — early
numbers suggest more like 0.2–1.1×. **Do not let anyone put 12–40× in the paper or
abstract.** Finish A4 to 144/144 and report the real number, whatever it is.

## When a job finishes

1. Write/refresh `results/<job>_summary.md` — honest answer to the question the ablation
   was designed to answer (see the original prompts in git history:
   `git log --oneline --all -- docs/DISPATCH_PACK.md` if you need the exact framing).
2. Do NOT edit `paper/*.tex` or `report/*.tex` yourself. Leave the summary ready for
   the integration pass.
3. Commit + push from this repo (you're on the same working directory as everyone else
   — `git pull --rebase` before pushing if `git status` shows you're behind).

## When the whole queue is done

Post a short status here or in STATUS.md's "what's left" section (read-only append,
don't rewrite the whole file) — do not attempt final paper integration yourself.
