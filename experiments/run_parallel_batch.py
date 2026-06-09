#!/usr/bin/env python3
"""Run fairness PGD experiments across datasets in parallel to separate files.

Each dataset writes to its own JSON file to avoid race conditions.
Results are merged at the end.
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_fairness_pgd import run_single_experiment


def run_dataset_batch(dataset, alphas, attacks, methods, n_seeds, device, output_path,
                      epochs=60, k_inner=5, pgd_steps=20):
    """Run all experiments for a single dataset, saving incrementally."""
    
    # Load existing progress
    all_results = []
    completed_keys = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            all_results = json.load(f)
        completed_keys = {
            (r['dataset'], r['alpha'], r['seed'], r['attack'], r['method'])
            for r in all_results
        }
        print(f"[{dataset}] Loaded {len(all_results)} existing results")
    
    # Build config list
    configs = []
    for alpha in alphas:
        for seed in range(n_seeds):
            for attack in attacks:
                for method in methods:
                    key = (dataset, alpha, seed, attack, method)
                    if key not in completed_keys:
                        configs.append((dataset, alpha, seed, attack, method))
    
    total = len(completed_keys) + len(configs)
    print(f"[{dataset}] {len(completed_keys)}/{total} already done, {len(configs)} remaining")
    
    def save():
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    for i, (ds, alpha, seed, attack, method) in enumerate(configs):
        label = f"[{dataset}] [{i+1}/{len(configs)}] {ds} α={alpha} seed={seed} attack={attack} method={method}"
        print(label, flush=True)
        try:
            t0 = time.time()
            result = run_single_experiment(
                ds, alpha, seed, attack, method,
                device=device, verbose=False,
                epochs=epochs, k_inner=k_inner, pgd_steps=pgd_steps
            )
            elapsed = time.time() - t0
            all_results.append(result)
            completed_keys.add((ds, alpha, seed, attack, method))
            save()
            print(f"  -> acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f"  -> FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    print(f"[{dataset}] Done. Total results: {len(all_results)}/{total}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--k_inner', type=int, default=5)
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--parallel', action='store_true', help='Run datasets in parallel subprocesses')
    args = parser.parse_args()
    
    if args.parallel:
        # Launch one subprocess per dataset
        procs = []
        for dataset in args.datasets:
            out_path = f'results/fairness_pgd_{dataset}.json'
            cmd = [
                sys.executable, __file__,
                '--datasets', dataset,
                '--alphas', *[str(a) for a in args.alphas],
                '--attacks', *args.attacks,
                '--methods', *args.methods,
                '--n_seeds', str(args.n_seeds),
                '--device', args.device,
                '--k_inner', str(args.k_inner),
                '--pgd_steps', str(args.pgd_steps),
                '--epochs', str(args.epochs),
            ]
            log_path = f'logs/batch_{dataset}.log'
            os.makedirs('logs', exist_ok=True)
            f = open(log_path, 'w')
            print(f"Launching {dataset} -> {out_path} (log: {log_path})")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append((dataset, p, f))
        
        print(f"\nLaunched {len(procs)} parallel jobs. Waiting for completion...")
        print(f"Tail logs: tail -f logs/batch_*.log")
        for dataset, p, f in procs:
            p.wait()
            f.close()
            print(f"[{dataset}] exited with code {p.returncode}")
        
        # Merge results
        print("\nMerging results...")
        merged = []
        for dataset in args.datasets:
            path = f'results/fairness_pgd_{dataset}.json'
            if os.path.exists(path):
                with open(path) as f:
                    merged.extend(json.load(f))
        
        merged_path = 'results/fairness_pgd_results.json'
        with open(merged_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Merged {len(merged)} results -> {merged_path}")
        
    else:
        # Single dataset mode (called by parallel launcher)
        for dataset in args.datasets:
            out_path = f'results/fairness_pgd_{dataset}.json'
            run_dataset_batch(
                dataset, args.alphas, args.attacks, args.methods, args.n_seeds,
                args.device, out_path,
                epochs=args.epochs, k_inner=args.k_inner, pgd_steps=args.pgd_steps
            )


if __name__ == '__main__':
    main()
