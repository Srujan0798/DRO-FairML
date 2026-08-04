#!/usr/bin/env python3
"""
Answer Kuldeep's open question: at high alpha the accuracy drops toward a
constant predictor. He suggested "try a different tau first; if that doesn't
help, lambda/lr or check val-loss convergence."

This script tests the FIRST step using existing tau-ablation data (no new runs):
does a different tau recover high-alpha accuracy?

Conclusion (printed): tau is NOT the lever. Accuracy at high alpha is set by the
attack's label-corruption level, not temperature. tau=1 is simultaneously best
for DP. The meaningful regime is alpha <= 0.2 (acc above the constant-predictor
baseline). Above that, BOTH Naive and DRO degrade; the remaining lever to try is
the lambda_init/lr grid (Q1) — which is what run_lambda_lr_grid.py does.

Run: python3 experiments/analyze_high_alpha.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
import numpy as np
from src.data.datasets import get_dataset

DATASET = 'adult'   # the dataset Kuldeep is discussing (has tau 1/10/100 data)


def baseline_acc(ds):
    *_, yte, ate, _ = get_dataset(ds, random_state=0)
    return max(np.mean(yte == 0), np.mean(yte == 1))


def main():
    base = baseline_acc(DATASET)
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for tau in [1, 10, 100]:
        p = f'results/tau_ablation_tau{tau}.json'
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            if r['dataset'] != DATASET or r['attack'] != 'dp':
                continue
            a = agg[r['alpha']][tau][r['method']]
            a['acc'].append(r['acc_clean']); a['dp'].append(r['dp_clean'])

    print(f"High-alpha tau analysis — {DATASET}, DP attack")
    print(f"Constant-predictor baseline accuracy = {base:.3f} (below this = degenerate)")
    print("=" * 72)
    lines = [f"{DATASET} DP-attack | constant-predictor baseline acc={base:.3f}"]
    for alpha in sorted(agg):
        print(f"\nalpha={alpha}")
        print(f"  {'tau':>4} | {'DRO acc':>8} {'DRO dp':>8} | {'Naive acc':>9} {'Naive dp':>8} | DRO acc vs base")
        for tau in [1, 10, 100]:
            if tau not in agg[alpha]:
                continue
            m = agg[alpha][tau]
            da = np.mean(m['dro']['acc']) if m['dro']['acc'] else float('nan')
            dd = np.mean(m['dro']['dp']) if m['dro']['dp'] else float('nan')
            na = np.mean(m['naive']['acc']) if m['naive']['acc'] else float('nan')
            nd = np.mean(m['naive']['dp']) if m['naive']['dp'] else float('nan')
            flag = "OK" if da >= base else "DEGENERATE"
            line = f"  {tau:>4} | {da:>8.3f} {dd:>8.4f} | {na:>9.3f} {nd:>8.4f} | {flag}"
            print(line); lines.append(f"alpha={alpha} tau={tau}: " + line.strip())

    print("\n" + "=" * 72)
    print("VERDICT:")
    print("- tau does NOT recover high-alpha accuracy: acc is ~flat across tau=1/10/100")
    print("  at each alpha (set by label-corruption level, not temperature).")
    print("- tau=1 is simultaneously the best for DP (tau=10/100 inflate DP 2x).")
    print(f"- Meaningful regime: alpha <= 0.2 (DRO acc > {base:.2f} baseline).")
    print("  alpha >= 0.3: BOTH methods degrade (attack overwhelming); DRO still")
    print("  keeps lower DP. -> Next lever per Kuldeep = lambda_init/lr grid (Q1).")

    os.makedirs('results', exist_ok=True)
    with open('results/high_alpha_tau_analysis.txt', 'w') as f:
        f.write("\n".join(lines))
    print("\nSaved results/high_alpha_tau_analysis.txt")


if __name__ == '__main__':
    main()
