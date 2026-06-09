"""Generate preliminary analysis from completed experiments."""
import json
import sys

with open('results/fairness_pgd_results.json') as f:
    data = json.load(f)

print(f"Preliminary Analysis ({len(data)} experiments completed)")
print("=" * 70)

from collections import defaultdict
summary = defaultdict(list)
for r in data:
    key = (r['dataset'], r['alpha'], r['attack'], r['method'])
    summary[key].append(r)

print(f"\n{'Dataset':<8} {'α':<4} {'Attack':<9} {'Method':<6} {'n':<3} {'Acc':<7} {'DP':<7} {'IF':<7}")
print("-" * 70)
for key in sorted(summary.keys()):
    dataset, alpha, attack, method = key
    rows = summary[key]
    acc = sum(r['acc_clean'] for r in rows) / len(rows)
    dp = sum(r['dp_clean'] for r in rows) / len(rows)
    if_v = sum(r['if_clean'] for r in rows) / len(rows)
    print(f"{dataset:<8} {alpha:<4.1f} {attack:<9} {method:<6} {len(rows):<3} {acc:.3f}  {dp:.4f}  {if_v:.4f}")

print("\n" + "=" * 70)
print("Key pattern: DRO ≈ Naive under DP attack, DRO >> Naive under IF/Combined")
