#!/usr/bin/env python3
"""Merge results from separate dataset runs into all_results.json."""

import os
import json
import pickle

def merge_results():
    all_results = []
    datasets = ['adult', 'credit', 'lsac']

    for ds in datasets:
        # Try checkpoint first
        ckpt_path = f'results/full_{ds}/checkpoint.pkl'
        final_path = f'results/full_{ds}/all_results.json'

        results = []
        if os.path.exists(final_path):
            with open(final_path, 'r') as f:
                results = json.load(f)
            print(f'{ds}: loaded {len(results)} from all_results.json')
        elif os.path.exists(ckpt_path):
            with open(ckpt_path, 'rb') as f:
                data = pickle.load(f)
            results = data.get('results', [])
            print(f'{ds}: loaded {len(results)} from checkpoint.pkl')

        all_results.extend(results)

    print(f'\nTotal results: {len(all_results)}')

    # Save merged
    os.makedirs('results', exist_ok=True)
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved to results/all_results.json')

    # Summary
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in all_results:
        key = (r['dataset'], r['alpha'])
        by_cell[key].append(r)

    print('\n=== SUMMARY ===')
    total_wins = 0
    for ds in ['adult', 'credit', 'lsac']:
        for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
            key = (ds, a)
            if key not in by_cell:
                continue
            sub = by_cell[key]
            n_dp = sum(r['naive']['clean']['dp_violation'] for r in sub) / len(sub)
            d_dp = sum(r['dro']['clean']['dp_violation'] for r in sub) / len(sub)
            win = d_dp < n_dp
            if win:
                total_wins += 1
            print(f'{ds} α={a}: {len(sub)}/10 | N_DP={n_dp:.4f} D_DP={d_dp:.4f} {"WIN" if win else "LOSS"}')

    print(f'\nTotal DP wins: {total_wins}/15 (datasets × alphas)')
    return all_results

if __name__ == '__main__':
    merge_results()