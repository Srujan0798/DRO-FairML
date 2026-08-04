#!/usr/bin/env python3
"""
Auto-pipeline: monitors running experiments, triggers next steps.
1. Waits for Credit alpha=0.4 to finish → launches lambda grid
2. Waits for kNN k=10 to finish
3. Waits for LSAC canonical to reach 180 rows → runs Wilcoxon, plots, tables
"""
import json, os, time, subprocess, sys

RESULTS = '/Users/srujansai/Desktop/DRO-FairML/results'
ROOT = '/Users/srujansai/Desktop/DRO-FairML'

def count_rows(file_path, dataset=None):
    if not os.path.exists(file_path):
        return 0
    try:
        d = json.load(open(file_path))
        if dataset:
            return sum(1 for r in d if r.get('dataset') == dataset)
        return len(d)
    except:
        return 0

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False

def run_cmd(cmd, desc):
    print(f"[PIPELINE] Running: {desc}")
    print(f"  CMD: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode}): {r.stderr[:500]}")
    else:
        print(f"  OK: {r.stdout[-300:] if r.stdout else '(no output)'}")
    return r.returncode == 0

PIDS = {'credit': 27103, 'lsac': 27100, 'knn': 23001}
lambda_launched = False
wilcoxon_done = False
plots_done = False

print("="*60)
print("AUTO-PIPELINE STARTED")
print("="*60)
print(f"PIDs: {PIDS}")

while True:
    now = time.strftime('%H:%M:%S')
    
    # Check Credit alpha=0.4
    credit_count = count_rows(f'{RESULTS}/canonical_tau1.json', 'credit')
    credit_running = is_running(PIDS['credit'])
    
    # Check LSAC
    lsac_count = count_rows(f'{RESULTS}/canonical_tau1.json', 'lsac')
    lsac_running = is_running(PIDS['lsac'])
    
    # Check kNN k=10
    knn_count = count_rows(f'{RESULTS}/knn_ablation_k10.json')
    knn_running = is_running(PIDS['knn'])
    
    total_canonical = count_rows(f'{RESULTS}/canonical_tau1.json')
    
    print(f"[{now}] Canonical: {total_canonical}/540 (LSAC: {lsac_count}/180, Credit: {credit_count}/180) | "
          f"kNN k=10: {knn_count}/144 | "
          f"Running: credit={credit_running} lsac={lsac_running} knn={knn_running}")
    
    # Step 1: When Credit finishes, launch lambda grid
    if not credit_running and not lambda_launched and credit_count >= 161:
        print(f"\n[PIPELINE] Credit finished ({credit_count}/180). Launching lambda grid...")
        subprocess.Popen(
            [sys.executable, '-u', 'experiments/run_lambda_grid_comprehensive.py',
             '--datasets', 'adult', 'credit',
             '--attacks', 'dp', 'if', 'combined',
             '--n_seeds', '3'],
            cwd=ROOT,
            stdout=open(f'{ROOT}/lambda_comprehensive.log', 'w'),
            stderr=subprocess.STDOUT
        )
        lambda_launched = True
        print("[PIPELINE] Lambda grid launched.")
    
    # Step 2: Run Wilcoxon when LSAC ≥120 or all done
    if not wilcoxon_done and lsac_count >= 120:
        print(f"\n[PIPELINE] LSAC ≥120 ({lsac_count}/180). Running Wilcoxon + basic plots...")
        run_cmd('python3 experiments/compute_canonical_wilcoxon.py', 'Wilcoxon')
        wilcoxon_done = True
    
    # Step 3: Generate all plots when canonical ≥400 or all done
    if not plots_done and (total_canonical >= 450 or (not lsac_running and lsac_count > 0)):
        print(f"\n[PIPELINE] Canonical sufficient ({total_canonical}/540). Generating plots...")
        run_cmd('python3 experiments/generate_final_figures.py', 'Final figures')
        plots_done = True
    
    # Step 4: Check if everything is done
    if (not credit_running and not lsac_running and not knn_running and 
        total_canonical >= 341 and knn_count >= 134):
        print(f"\n[PIPELINE] ALL PROCESSES DONE.")
        print(f"  Canonical: {total_canonical}/540")
        print(f"  kNN k=10: {knn_count}/144")
        
        # Final pipeline
        if not wilcoxon_done:
            run_cmd('python3 experiments/compute_canonical_wilcoxon.py', 'Wilcoxon (final)')
        if not plots_done:
            run_cmd('python3 experiments/generate_final_figures.py', 'Final figures (final)')
        
        print("\n[PIPELINE] ALL DONE. Exiting.")
        break
    
    # Sleep 60s between checks
    time.sleep(60)
