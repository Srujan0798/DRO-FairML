"""Check experiment progress and show partial results."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from glob import glob


def check_progress(results_dir='results'):
    """Check how many experiments have completed and show partial results."""
    
    # Try checkpoint first
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pkl')
    json_path = os.path.join(results_dir, 'all_results.json')
    
    results = []
    if os.path.exists(checkpoint_path):
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        results = checkpoint['results']
        print(f"Found checkpoint with {len(results)} experiments")
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
        print(f"Found final results with {len(results)} experiments")
    else:
        print("No results found yet.")
        return
    
    # Group by dataset and alpha
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r['dataset'], r['alpha'])
        grouped[key].append(r)
    
    print("\n" + "="*80)
    print("PARTIAL RESULTS")
    print("="*80)
    
    for (dataset, alpha), runs in sorted(grouped.items()):
        print(f"\n{dataset.upper()} α={alpha} ({len(runs)} seeds)")
        print("-"*60)
        
        for method in ['naive', 'dro']:
            accs = [r[method]['accuracy'] for r in runs]
            dps = [r[method]['dp_violation'] for r in runs]
            ifs = [r[method]['if_violation'] for r in runs]
            
            se_acc = np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0
            se_dp = np.std(dps) / np.sqrt(len(dps)) if len(dps) > 1 else 0
            se_if = np.std(ifs) / np.sqrt(len(ifs)) if len(ifs) > 1 else 0
            
            print(f"  {method.upper():8s}: Acc={np.mean(accs):.4f}±{se_acc:.4f}, "
                  f"DP={np.mean(dps):.4f}±{se_dp:.4f}, IF={np.mean(ifs):.4f}±{se_if:.4f}")
    
    # Progress summary
    total_expected = 3 * 5 * 10 * 2  # datasets * alphas * seeds * methods
    total_completed = len(results)
    print(f"\n{'='*80}")
    print(f"Progress: {total_completed} experiments completed")
    print(f"Expected: ~150 experiments (3 datasets × 5 alphas × 10 seeds)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    check_progress()
