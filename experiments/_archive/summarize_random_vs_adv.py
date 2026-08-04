#!/usr/bin/env python3
"""Summarize random vs adversarial comparison results."""
import json
from collections import defaultdict
import numpy as np

with open('results/random_vs_adversarial_new.json') as f:
    results = json.load(f)

print("=" * 70)
print("Random vs Adversarial Corruption Comparison (27 runs)")
print("=" * 70)

groups = defaultdict(list)
for r in results:
    groups[(r['dataset'], r['alpha'])].append(r)

for (ds, alpha), entries in sorted(groups.items()):
    print(f"\n{ds.upper()} α={alpha:.1f}:")
    
    clean_dps = [e['clean']['dp'] for e in entries]
    rand_dps = [e['random']['dp'] for e in entries]
    adv_dps = [e['adversarial']['dp'] for e in entries]
    
    print(f"  Clean:     {np.mean(clean_dps):.4f}")
    print(f"  Random:    {np.mean(rand_dps):.4f} (Δ = {np.mean(rand_dps)-np.mean(clean_dps):+.4f})")
    print(f"  Adversarial: {np.mean(adv_dps):.4f} (Δ = {np.mean(adv_dps)-np.mean(clean_dps):+.4f})")
    
    # Count how many times adversarial is worse than random
    adv_worse = sum(1 for e in entries if e['adversarial']['dp'] > e['random']['dp'])
    print(f"  Adversarial worse than random: {adv_worse}/{len(entries)} seeds")
    
    # For cases where random increased DP, compute ratio
    ratios = []
    for e in entries:
        dp_rand_delta = e['random']['dp'] - e['clean']['dp']
        dp_adv_delta = e['adversarial']['dp'] - e['clean']['dp']
        if dp_rand_delta > 0.001:  # Only when random actually increased DP
            ratios.append(dp_adv_delta / dp_rand_delta)
    if ratios:
        print(f"  Adv/Random ratio (when random increases DP): {np.mean(ratios):.1f}x (range: {min(ratios):.1f}-{max(ratios):.1f}x)")

print("\n" + "=" * 70)
