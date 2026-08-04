# Agent coordination (2026-08-04) — data safety

**Source of truth for live sequencing while the IF sweep is open.**

## The one hard rule

**Only the IF sweep (`experiments/run_if_parallel.py`, pid 10146) may write `results/canonical_tau1.json` until total=540 / IF=180.**

That driver rewrites the whole file atomically after each config. Any second writer (`run_fairness_pgd.py`, `run_canonical.py`, another parallel driver, accidental regen that re-runs experiments) can clobber **all three attacks**.

| Process | May write `canonical_tau1.json`? |
|---------|----------------------------------|
| pid 10146 `run_if_parallel.py` | **YES — sole writer** |
| Agent H finalize | **NO** until gate 540/180; then **read-only** on JSON (regens tables/figs/PDFs only) |
| Agents J / K / L / M | **NEVER** write canonical |
| Monitors that regen | **Down** until H owns finalize (do not re-arm competing monitors) |

## Agent sequencing

| Agent | Can start | Depends on | Must not |
|-------|-----------|------------|----------|
| **Sweep** | running | — | — |
| **H** (finalize + real IF numbers) | when total=540 and IF=180 | sweep done | start any experiment run; poll only until ready |
| **K** (paper/report) | after H commits final tables/PDFs | H | invent IF numbers; touch canonical |
| **J** (repo cleanup) | now | independent | regenerate figures/PDFs while H owns that; prefer code/docs/Makefile first |
| **L** (verify everything) | last | H + K + J | ship with unresolved mismatches |
| **M** (UTKFace) | when download/GPU ready | fully independent | write canonical; report synthetic as real |

### J ↔ H figure/PDF conflict

Simplest fix: **J does code/docs/Makefile/fail-loud first and leaves figure regeneration to H.**

## Agent H contract

1. **Do not start any experiment run.**
2. `scripts/agent_h_finalize.sh` **polls** (`--wait`) until gate passes; does not generate data.
3. After gate: Wilcoxon, tables, `make results/deliverables/paper/report`, `results/if_wilcoxon_summary.txt`.
4. Judgment pass (prose, meeting brief) is separate; no auto-commit required by finalize.

## Read-only status signal

```bash
# sole writer check + progress (no writes)
python3 -c "import json,collections;d=json.load(open('results/canonical_tau1.json'));c=collections.Counter(r['attack'] for r in d);print(len(d),dict(c))"
ps -p 10146 -o pid,etime,command   # should be the only experiment parent on this file
./scripts/watch_sweep_readonly.sh  # polls; exits when IF=180; zero file writes
```

## Clear-to-finalize signal

When all of these hold, agents may proceed with H:

1. `ps -p 10146` is gone **or** log says DONE with IF=180
2. `total==540` and `if==180` in canonical
3. unique `(dataset,attack,alpha,seed,method)` count == 540 (no duplicate corruption)
4. `max |if_clean|` for `attack==if` ≫ 1e-6
5. No other process has the file open for write

Then: `./scripts/agent_h_finalize.sh` (or `--wait` then full run).
