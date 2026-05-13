# Project Completion Plan — DRO-FAIR

**Goal:** Complete empirical record, paper artifacts, and defensive items for submission.

---

## Stream A — Empirical (150-run is gate)

| # | Task | Command | Output | Depends |
|---|------|---------|--------|---------|
| A1 | Run experiments | `python3 experiments/run_experiments.py` | `results/all_results.json` (150 entries) | — |
| A2 | Verify completeness | Check no missing seeds in checkpoint | Same JSON | A1 |
| A3 | Generate results | `python3 experiments/generate_results.py` | `summary_stats.csv`, `table1_latex.tex`, `table1_results.csv`, figures | A2 |
| A4 | Spot-check | Run validation snippet (see below) | PASS/FAIL | A3 |
| A5 | Ablations | `python3 experiments/run_ablations.py` | `results/ablation_full.json` | — (parallel) |
| A6 | Random vs adversarial | `python3 experiments/run_random_vs_adversarial.py` | `results/random_vs_adversarial.json` | — (parallel) |
| A7 | Theory verification | `python3 experiments/verify_theory.py` | Pass/fail + plot | — (parallel) |

## Stream B — Scientific (characterize Adult regression)

| # | Task | Command | Output | Depends |
|---|------|---------|--------|---------|
| B1 | λ trajectory diagnostic | `python3 scripts/diagnose_lambda.py adult 0.2 42` | 3 PNGs + CSVs | A2 |
| B2 | Group-rate comparison | `python3 scripts/compare_group_rates.py` | `figures/group_rates_adult_vs_lsac.png` | A2 |
| B3 | λ_max sensitivity | `python3 scripts/lambda_max_sweep.py` | CSV + conclusion | — |
| B4 | λ_0=0 test | `python3 scripts/lambda0_sweep.py` | CSV + verdict | — |

## Stream C — Paper artifacts

| # | Task | Command | Output | Depends |
|---|------|---------|--------|---------|
| C1 | Table 1 LaTeX | In generate_results.py | `paper/table1.tex` | A3 |
| C2 | Figure 1 | In generate_results.py | `paper/figures/main_results.pdf` | A3 |
| C3 | Figure 2 (ablation) | In generate_results.py | `paper/figures/ablation.pdf` | A5 |
| C4 | Figure 3 (adv vs random) | In generate_results.py | `paper/figures/adv_vs_random.pdf` | A6 |
| C5 | Figure 4 (λ + group rates) | `python3 scripts/make_discussion_figure.py` | `paper/figures/adult_mechanism.pdf` | B1, B2 |
| C6 | Sensitivity table | `python3 scripts/hyperparam_sensitivity.py` | `results/sensitivity.csv` | — |
| C7 | Runtime table | In generate_results.py | `paper/table_runtime.tex` | A2 |
| C8 | Theory figure | `python3 scripts/make_theory_figure.py` | `paper/figures/theorem_check.pdf` | A7 |

## Stream D — Writeup

| # | Task | Output | Depends |
|---|------|--------|---------|
| D1 | Discussion: Adult over-correction | `paper/discussion.tex` | B1, B2 |
| D2 | Limitations paragraph | Same file | — |
| D3 | Runtime paragraph | Same file | C7 |
| D4 | Threats-to-validity | Same file | — |

## Stream E — Defensive

| # | Task | Command | Depends |
|---|------|--------|---------|
| E1 | Config default fate | Decide + commit | — |
| E2 | Fix fit() docstring | One commit | — |
| E3 | Fix UserWarning in diagnose_lambda.py | One commit | — |
| E4 | README "How to reproduce" | Updated README | A4 |
| E5 | LIMITATIONS.md | New file | D2 |
| E6 | Pin requirements.txt | Updated file | — |
| E7 | Tag submission | `git tag submission-camera-ready` | All |

---

## Validation Snippet (A4)

```bash
python3 << 'PYEOF'
import json, numpy as np
results = json.load(open('results/all_results.json'))
print(f"Total: {len(results)} experiments")
wins = {"dp": 0, "if": 0, "total": 0}
cells = []
for ds in ['adult', 'credit', 'lsac']:
    for a in [0.1, 0.2, 0.3]:
        sub = [r for r in results if r['dataset'] == ds and r['alpha'] == a]
        if not sub:
            continue
        n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in sub])
        d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in sub])
        n_if = np.mean([r['naive']['clean']['if_violation'] for r in sub])
        d_if = np.mean([r['dro']['clean']['if_violation'] for r in sub])
        n_acc = np.mean([r['naive']['clean']['accuracy'] for r in sub])
        d_acc = np.mean([r['dro']['clean']['accuracy'] for r in sub])
        dp_win = d_dp < n_dp
        if_win = d_if < n_if
        if dp_win: wins["dp"] += 1
        if if_win: wins["if"] += 1
        wins["total"] += 1
        print(f"{ds} a={a}: DP {n_dp:.4f}->{d_dp:.4f} {'WIN' if dp_win else 'LOSS'}, IF {n_if:.4f}->{d_if:.4f} {'WIN' if if_win else 'LOSS'}, Acc {n_acc:.4f}->{d_acc:.4f}")

        # Credit α=0.4 special check
        sub04 = [r for r in results if r['dataset'] == 'credit' and r['alpha'] == 0.4]
        if sub04:
            d_acc04 = np.mean([r['dro']['clean']['accuracy'] for r in sub04])
            print(f"  Credit α=0.4 DRO acc: {d_acc04:.4f} {'OK' if d_acc04 >= 0.60 else 'FAIL'}")

print(f"\nDP wins: {wins['dp']}/9")
print(f"IF wins: {wins['if']}/9")
print(f"RESULT: {'PASS' if wins['dp'] >= 6 else 'FAIL'}")
PYEOF
```

---

## Critical Path

```
Agent 1: A1 (wait) → A2 → A3 → A4 → A5 (parallel to A1) → A6 (parallel to A1) → A7 (parallel to A1)
Agent 2: E1, E2, E3, E6 (parallel, fast) → B1, B2 → D1, D2, D3, D4 → C1, C2, C5, C7 → E4, E5, E7
```

---

## Riskiest Assumption

In-progress 150-run produces DP wins ≥ 6/9, Credit α=0.4 acc ≥ 0.70. If A4 shows red, escalate to B3/B4.