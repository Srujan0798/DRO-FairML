#!/usr/bin/env python3
"""Targeted K_inner=10 comparison for critical Adult configs."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_fairness_pgd import run_single_experiment


def main():
    os.makedirs('results/k10_comparison', exist_ok=True)
    
    # Target the critical configs where K=5 showed DRO wins at alpha=0.4
    configs = [
        ('adult', 0.4, 0, 'dp', 'dro'),
        ('adult', 0.4, 1, 'dp', 'dro'),
        ('adult', 0.4, 2, 'dp', 'dro'),
        ('adult', 0.4, 0, 'combined', 'dro'),
        ('adult', 0.4, 1, 'combined', 'dro'),
        ('adult', 0.4, 2, 'combined', 'dro'),
    ]
    
    results = []
    out_path = 'results/k10_comparison/adult_alpha04_k10.json'
    
    for i, (ds, alpha, seed, attack, method) in enumerate(configs):
        print(f'[{i+1}/{len(configs)}] {ds} α={alpha} s={seed} {attack} {method} k_inner=10')
        try:
            r = run_single_experiment(ds, alpha, seed, attack, method,
                                       device='cpu', verbose=False,
                                       epochs=60, k_inner=10, pgd_steps=20)
            results.append(r)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'  -> acc={r["acc_clean"]:.3f} dp={r["dp_clean"]:.4f} time={r.get("total_time", 0):.0f}s')
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()
    
    print(f'\nSaved {len(results)} K=10 results to {out_path}')


if __name__ == '__main__':
    main()
