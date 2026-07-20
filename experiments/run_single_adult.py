#!/usr/bin/env python3
"""Run a single adult experiment and save to a temp file."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_fairness_pgd import run_single_experiment

def main():
    alpha = float(sys.argv[1])
    seed = int(sys.argv[2])
    attack = sys.argv[3]
    method = sys.argv[4]
    out_file = sys.argv[5]
    
    result = run_single_experiment(
        'adult', alpha, seed, attack, method,
        device='cpu', verbose=False,
        epochs=60, k_inner=10, pgd_steps=20
    )
    
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"DONE: adult α={alpha} seed={seed} attack={attack} method={method}")
    print(f"  acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} if={result['if_clean']:.4f}")

if __name__ == '__main__':
    main()
