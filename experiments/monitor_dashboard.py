#!/usr/bin/env python3
"""Real-time experiment progress dashboard."""
import pickle, os, sys, time
from collections import defaultdict

DATASETS = ['adult', 'credit', 'lsac']
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]

def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def format_results(data):
    if not data:
        return "  (no data)"
    lines = []
    by_alpha = defaultdict(list)
    for r in data['results']:
        by_alpha[r['alpha']].append(r)
    for alpha in sorted(by_alpha):
        rr = by_alpha[alpha]
        wins = sum(1 for r in rr if r['dro']['clean']['dp_violation'] < r['naive']['clean']['dp_violation'])
        lines.append(f"    α={alpha}: {len(rr):2d} seeds, DRO wins {wins}/{len(rr)}")
    return "\n".join(lines)

def main():
    os.system('clear' if os.name != 'nt' else 'cls')
    print("=" * 70)
    print(f"  DRO-FAIR Experiment Dashboard  —  {time.strftime('%H:%M:%S')}")
    print("=" * 70)

    total_done = 0
    total_wins = 0
    total_seeds = 0

    # Adult (parallel-by-alpha)
    print(f"\n[ADULT] (parallel by alpha)")
    adult_done = 0
    adult_wins = 0
    adult_total = 0
    for alpha in ALPHAS:
        cp = load_checkpoint(f'results/adult_a{alpha}/checkpoint.pkl')
        if cp:
            n = len(cp['completed_keys'])
            wins = sum(1 for r in cp['results'] if r['dro']['clean']['dp_violation'] < r['naive']['clean']['dp_violation'])
            adult_done += n
            adult_wins += wins
            adult_total += len(cp['results'])
            print(f"  α={alpha}: {n:2d}/10 seeds, DRO wins {wins}/{n}")
    if adult_total == 0:
        print("  No checkpoints yet")
    pct = adult_done / 50 * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  Total: [{bar}] {adult_done}/50 ({pct:.1f}%), wins={adult_wins}/{adult_total}")
    total_done += adult_done
    total_wins += adult_wins
    total_seeds += adult_total

    # Credit and LSAC (single-process)
    for ds in ['credit', 'lsac']:
        print(f"\n[{ds.upper()}]")
        cp = load_checkpoint(f'results/full_{ds}/checkpoint.pkl')
        if cp:
            n = len(cp['completed_keys'])
            wins = sum(1 for r in cp['results'] if r['dro']['clean']['dp_violation'] < r['naive']['clean']['dp_violation'])
            total_done += n
            total_wins += wins
            total_seeds += len(cp['results'])
            pct = n / 50 * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  Progress: [{bar}] {n}/50 ({pct:.1f}%)")
            print(f"  DRO wins: {wins}/{n}")
            print(format_results(cp))
        else:
            print("  No checkpoint yet")

    print(f"\n{'='*70}")
    print(f"  TOTAL: {total_done}/150 seeds ({100*total_done/150:.1f}%)")
    pct = 100*total_wins/total_seeds if total_seeds > 0 else 0
    print(f"  DRO wins: {total_wins}/{total_seeds} ({pct:.0f}% of evaluated)")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
