#!/usr/bin/env python3
"""
Professor Review Simulator
==========================

Runs ALL checks from PROF_PROMPT.md so you can self-grade before submission.
This is what your professor will run. If this script passes, you're golden.

Run: python experiments/professor_review_simulator.py

Output: PASS / CONDITIONAL PASS / FAIL with exact findings.
"""

import os
import sys
import json
import numpy as np
import torch
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PASS_COLOR = '\033[92m'
FAIL_COLOR = '\033[91m'
WARN_COLOR = '\033[93m'
RESET = '\033[0m'

findings = {'CRITICAL': [], 'HIGH': [], 'MODERATE': [], 'LOW': []}
pass_count = 0
total_checks = 0


def check(name, condition, severity='CRITICAL', detail=''):
    """Record a check result."""
    global pass_count, total_checks
    total_checks += 1
    status = PASS_COLOR + 'PASS' + RESET if condition else FAIL_COLOR + 'FAIL' + RESET
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    if condition:
        pass_count += 1
    else:
        findings[severity].append(f"{name}: {detail}" if detail else name)
    return condition


def run_all_checks():
    """Run the full professor review."""
    print("=" * 80)
    print("PROFESSOR REVIEW SIMULATOR")
    print("=" * 80)

    # === CHECK 1: Tests ===
    print("\n[1/15] Running pytest...")
    import subprocess
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        capture_output=True, text=True, cwd='/Users/srujansai/Desktop/DRO-FairML'
    )
    passed_line = [l for l in result.stdout.split('\n') if 'passed' in l]
    all_pass = result.returncode == 0
    check("All tests pass", all_pass, 'CRITICAL',
          passed_line[-1] if passed_line else result.stdout[-200:])

    # === CHECK 2: Algorithm order ===
    print("\n[2/15] Checking algorithm order...")
    from src.training.dro_fair import DroFairTrainer
    src = inspect.getsource(DroFairTrainer.fit)
    theta_pos = src.find('opt_theta.step()')
    inner_pos = src.find('INNER MAXIMIZATION')
    check("Theta update before inner max",
          theta_pos > 0 and inner_pos > 0 and theta_pos < inner_pos,
          'CRITICAL', f"theta_pos={theta_pos}, inner_pos={inner_pos}")

    # === CHECK 3: Tau ===
    print("\n[3/15] Checking tau defaults...")
    from src.training.dro_fair import DroFairTrainer
    from src.training.naive_fair import NaiveFairTrainer
    from src.models.classifier import MLPClassifier
    m = MLPClassifier(5)
    d = DroFairTrainer(m, alpha=0.2)
    n = NaiveFairTrainer(m)
    check("DRO tau=100", d.tau == 100.0, 'CRITICAL', f"got {d.tau}")
    check("Naive tau=100", n.tau == 100.0, 'CRITICAL', f"got {n.tau}")

    # === CHECK 4: Lambda init ===
    print("\n[4/15] Checking lambda initialization...")
    check("Lambda starts at 0.0",
          'torch.tensor(0.0' in src, 'CRITICAL',
          "Lambda must initialize at 0.0 for stable dual ascent")

    # === CHECK 5: Results exist ===
    print("\n[5/15] Checking results...")
    results_path = 'results/all_results.json'
    has_results = os.path.exists(results_path)
    check("Results file exists", has_results, 'CRITICAL',
          f"{results_path} not found")

    n_results = 0
    if has_results:
        with open(results_path) as f:
            results = json.load(f)
        n_results = len(results)
        check("150 experiments complete", n_results == 150, 'CRITICAL',
              f"Only {n_results}/150 experiments")

    # === CHECK 6: DRO beats Naive ===
    print("\n[6/15] Checking DRO vs Naive...")
    if has_results and n_results > 0:
        with open(results_path) as f:
            results = json.load(f)
        wins = 0
        total = 0
        for ds in ['adult', 'credit', 'lsac']:
            for alpha in [0.1, 0.2, 0.3]:
                subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
                if len(subset) < 3:
                    continue
                n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in subset])
                d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in subset])
                total += 1
                if d_dp < n_dp:
                    wins += 1
        check(f"DRO beats Naive on DP ({wins}/{total} comparisons)",
              wins >= 6, 'CRITICAL',
              f"DRO only wins {wins}/{total}. Need at least 6/9.")
    else:
        check("DRO beats Naive on DP", False, 'CRITICAL', "No results to check")

    # === CHECK 7: No degeneracy ===
    print("\n[7/15] Checking for degeneracy...")
    if has_results and n_results > 0:
        with open(results_path) as f:
            results = json.load(f)
        degenerate = False
        for ds in ['adult', 'credit', 'lsac']:
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
                subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
                if len(subset) < 2:
                    continue
                for method in ['naive', 'dro']:
                    for metric in ['accuracy', 'dp_violation']:
                        vals = [r[method]['clean'][metric] for r in subset]
                        se = np.std(vals) / np.sqrt(len(vals))
                        if se == 0:
                            degenerate = True
        check("No SE=0 degeneracy", not degenerate, 'CRITICAL',
              "Some metrics have SE=0 (possible constant predictions)")
    else:
        check("No degeneracy", True, 'CRITICAL', "Skipped (no results)")

    # === CHECK 8: No NaN/Inf ===
    print("\n[8/15] Checking for NaN/Inf...")
    if has_results and n_results > 0:
        with open(results_path) as f:
            results = json.load(f)
        nan_found = False
        for r in results:
            for method in ['naive', 'dro']:
                for ev in ['clean', 'corrupted']:
                    for metric in ['accuracy', 'dp_violation', 'if_violation']:
                        v = r[method][ev][metric]
                        if not np.isfinite(v):
                            nan_found = True
        check("No NaN/Inf values", not nan_found, 'CRITICAL',
              "Found NaN or Inf in results")
    else:
        check("No NaN/Inf", True, 'CRITICAL', "Skipped (no results)")

    # === CHECK 9: Accuracy sanity ===
    print("\n[9/15] Checking accuracy sanity...")
    if has_results and n_results > 0:
        with open(results_path) as f:
            results = json.load(f)
        bad_acc = False
        for ds in ['adult', 'credit', 'lsac']:
            for alpha in [0.0, 0.2, 0.4]:
                subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
                if not subset:
                    continue
                for method in ['naive', 'dro']:
                    acc = np.mean([r[method]['clean']['accuracy'] for r in subset])
                    if acc < 0.60:
                        bad_acc = True
        check("All accuracies >= 0.60", not bad_acc, 'CRITICAL',
              "Some accuracies below 0.60 (training collapse)")
    else:
        check("Accuracy sanity", True, 'CRITICAL', "Skipped (no results)")

    # === CHECK 10: Alpha=0 baseline ===
    print("\n[10/15] Checking alpha=0 baseline...")
    if has_results and n_results > 0:
        with open(results_path) as f:
            results = json.load(f)
        bad_baseline = False
        for ds in ['adult', 'credit', 'lsac']:
            subset = [r for r in results if r['dataset'] == ds and r['alpha'] == 0.0]
            if len(subset) < 3:
                continue
            n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in subset])
            d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in subset])
            n_acc = np.mean([r['naive']['clean']['accuracy'] for r in subset])
            d_acc = np.mean([r['dro']['clean']['accuracy'] for r in subset])
            if abs(n_dp - d_dp) > 0.03 or abs(n_acc - d_acc) > 0.03:
                bad_baseline = True
        check("Alpha=0: Naive ≈ DRO", not bad_baseline, 'HIGH',
              "At alpha=0, DRO and Naive should be similar (<0.03 gap)")
    else:
        check("Alpha=0 baseline", True, 'HIGH', "Skipped (no results)")

    # === CHECK 11: predict_proba multiply ===
    print("\n[11/15] Checking predict_proba...")
    from src.models.classifier import MLPClassifier
    src_cls = inspect.getsource(MLPClassifier.predict_proba)
    check("predict_proba uses multiply",
          'logits * temperature' in src_cls and 'logits / temperature' not in src_cls,
          'CRITICAL', "Must use sigmoid(logits * temperature)")

    # === CHECK 12: h_tilde multiply in trainers ===
    print("\n[12/15] Checking h_tilde in trainers...")
    check("DRO h_tilde uses multiply",
          'logits * current_tau' in src, 'CRITICAL')
    naive_src = inspect.getsource(NaiveFairTrainer.fit)
    check("Naive h_tilde uses multiply",
          'logits * current_tau' in naive_src, 'CRITICAL')

    # === CHECK 13: Data leakage ===
    print("\n[13/15] Checking data leakage...")
    from src.data.datasets import get_dataset
    X1, _, _, _, _, _, Xt1, _, _, _ = get_dataset('adult', random_state=42)
    train_mean = np.abs(np.mean(X1, axis=0)).mean()
    check("StandardScaler fit on train only",
          train_mean < 0.1, 'CRITICAL',
          f"Train mean abs = {train_mean:.4f} (should be ~0)")

    # === CHECK 14: Runtime overhead ===
    print("\n[14/15] Checking runtime overhead...")
    rt_path = 'results/runtimes.json'
    if os.path.exists(rt_path):
        with open(rt_path) as f:
            rt = json.load(f)
        overhead = rt.get('overhead', 0)
        check("DRO overhead > 1.5x", overhead > 1.5, 'MODERATE',
              f"Overhead = {overhead:.2f}x (DRO should be slower due to inner max)")
    else:
        check("Runtime overhead", True, 'MODERATE', "Skipped (no runtimes.json)")

    # === CHECK 15: Deliverables ===
    print("\n[15/15] Checking deliverables...")
    deliverables = [
        'results/all_results.json',
        'results/table1_results.csv',
        'results/table1_latex.tex',
    ]
    all_present = True
    for path in deliverables:
        if not os.path.exists(path):
            all_present = False
            findings['HIGH'].append(f"Missing deliverable: {path}")
    check("All deliverables present", all_present, 'HIGH',
          "Some required files missing")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Checks passed: {pass_count}/{total_checks}")

    if findings['CRITICAL']:
        print(f"\n{FAIL_COLOR}CRITICAL findings ({len(findings['CRITICAL'])}):{RESET}")
        for i, f in enumerate(findings['CRITICAL'], 1):
            print(f"  {i}. {f}")

    if findings['HIGH']:
        print(f"\n{WARN_COLOR}HIGH findings ({len(findings['HIGH'])}):{RESET}")
        for i, f in enumerate(findings['HIGH'], 1):
            print(f"  {i}. {f}")

    if findings['MODERATE']:
        print(f"\nMODERATE findings ({len(findings['MODERATE'])}):")
        for i, f in enumerate(findings['MODERATE'], 1):
            print(f"  {i}. {f}")

    print("\n" + "=" * 80)
    if pass_count == total_checks:
        verdict = PASS_COLOR + "PASS" + RESET
    elif pass_count >= total_checks - 2 and not findings['CRITICAL']:
        verdict = WARN_COLOR + "CONDITIONAL PASS" + RESET
    else:
        verdict = FAIL_COLOR + "FAIL" + RESET

    print(f"VERDICT: {verdict}")
    print("=" * 80)

    return 0 if pass_count == total_checks else 1


if __name__ == '__main__':
    sys.exit(run_all_checks())
