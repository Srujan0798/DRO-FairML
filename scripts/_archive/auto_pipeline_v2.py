#!/usr/bin/env python3
"""
Auto-pipeline v2: monitors experiments, triggers next steps.
Key fix: Credit and LSAC must NOT run simultaneously (same output file race).
Order: LSAC finishes → Credit remaining → lambda grid → Wilcoxon → plots → done.
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
    print(f"[PIPELINE] Running: {desc}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode}): {r.stderr[:500]}", flush=True)
    else:
        print(f"  OK", flush=True)
    return r.returncode == 0

LSAC_PID = 27100
KNN_PID = 23001

credit_launched = False
credit_pid = None
lambda_launched = False
wilcoxon_done = False
plots_done = False
tables_done = False

print("="*60, flush=True)
print("AUTO-PIPELINE v2 STARTED (serialized Credit/LSAC)", flush=True)
print("="*60, flush=True)

while True:
    now = time.strftime('%H:%M:%S')
    
    total_canonical = count_rows(f'{RESULTS}/canonical_tau1.json')
    credit_count = count_rows(f'{RESULTS}/canonical_tau1.json', 'credit')
    lsac_count = count_rows(f'{RESULTS}/canonical_tau1.json', 'lsac')
    knn_count = count_rows(f'{RESULTS}/knn_ablation_k10.json')
    
    lsac_running = is_running(LSAC_PID)
    knn_running = is_running(KNN_PID)
    credit_running = is_running(credit_pid) if credit_pid else False
    
    print(f"[{now}] Canon: {total_canonical}/540 (LSAC:{lsac_count}/180 Credit:{credit_count}/180) "
          f"kNN:{knn_count}/144 | LSAC={lsac_running} Credit={credit_running} kNN={knn_running}", flush=True)
    
    # Step 1: When LSAC finishes, launch Credit alpha=0.4
    if not credit_launched and not lsac_running and lsac_count > 0:
        print(f"\n[PIPELINE] LSAC done ({lsac_count}/180). Launching Credit alpha=0.4...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, '-u', 'experiments/run_canonical.py',
             '--datasets', 'credit', '--alphas', '0.4',
             '--n_seeds', '6', '--epochs', '60', '--k_inner', '10', '--pgd_steps', '20'],
            cwd=ROOT,
            stdout=open(f'{ROOT}/credit_alpha04_v2.log', 'w'),
            stderr=subprocess.STDOUT
        )
        credit_pid = proc.pid
        credit_launched = True
        print(f"[PIPELINE] Credit launched (PID={credit_pid})", flush=True)
    
    # Step 2: When Credit finishes, launch lambda grid
    if credit_launched and not credit_running and not lambda_launched:
        print(f"\n[PIPELINE] Credit done ({credit_count}/180). Launching lambda grid...", flush=True)
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
        print("[PIPELINE] Lambda grid launched.", flush=True)
    
    # Step 3: Wilcoxon when enough canonical data
    if not wilcoxon_done and total_canonical >= 450:
        print(f"\n[PIPELINE] Running Wilcoxon ({total_canonical} rows)...", flush=True)
        run_cmd('python3 experiments/compute_canonical_wilcoxon.py', 'Wilcoxon')
        wilcoxon_done = True
    
    # Step 4: Final plots
    if not plots_done and total_canonical >= 450:
        print(f"\n[PIPELINE] Generating final plots...", flush=True)
        run_cmd('python3 experiments/generate_final_figures.py', 'Final figures')
        plots_done = True
    
    # Step 5: Tables
    if not tables_done and total_canonical >= 450:
        print(f"\n[PIPELINE] Generating tables...", flush=True)
        # Copy results to meeting folder
        run_cmd('cp results/canonical_wilcoxon.md kuldeep_meeting/', 'Copy wilcoxon')
        run_cmd('cp results/canonical_wilcoxon.csv kuldeep_meeting/', 'Copy wilcoxon csv')
        tables_done = True
    
    # Step 6: Check if ALL done
    all_canon_done = (not lsac_running) and (not credit_running or not credit_launched)
    credit_done = credit_launched and not credit_running
    
    if (all_canon_done and credit_done and knn_count >= 134 and 
        not lsac_running and total_canonical >= 341):
        print(f"\n[PIPELINE] === ALL EXPERIMENTS DONE ===", flush=True)
        print(f"  Canonical: {total_canonical}/540", flush=True)
        print(f"  kNN k=10: {knn_count}/144", flush=True)
        
        if not wilcoxon_done:
            run_cmd('python3 experiments/compute_canonical_wilcoxon.py', 'Wilcoxon (final)')
        if not plots_done:
            run_cmd('python3 experiments/generate_final_figures.py', 'Final figures (final)')
        
        print("\n[PIPELINE] ALL DONE. Exiting.", flush=True)
        break
    
    time.sleep(60)
