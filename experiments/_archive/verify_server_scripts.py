#!/usr/bin/env python3
"""
Verify all UTKFace server scripts parse, import, and accept their args.
Does NOT run any training. Intended to be run on flair2.iitgn.ac.in
immediately after `git pull` before queuing the long GPU jobs.

Usage:
    venv/bin/python3 experiments/verify_server_scripts.py
"""
import ast
import importlib
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCRIPTS = [
    'experiments/run_lambda_diagnostic.py',
    'experiments/run_utkface_extended.py',
    'experiments/run_utkface_pixel_pgd.py',
    'experiments/run_utkface_randinit.py',
    'experiments/plot_lambda_diagnostic.py',
]


def main():
    fail = 0
    for s in SCRIPTS:
        path = os.path.join(ROOT, s)
        try:
            ast.parse(open(path).read())
            print(f"PARSE OK   {s}")
        except SyntaxError as e:
            print(f"PARSE FAIL {s}: {e}")
            fail += 1
            continue
        # spawn with --help to confirm argparse wiring + imports
        try:
            r = subprocess.run([sys.executable, path, '--help'],
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                print(f"HELP FAIL  {s}: exit={r.returncode}")
                print(r.stderr.splitlines()[-5:])
                fail += 1
            else:
                print(f"HELP OK    {s}")
        except subprocess.TimeoutExpired:
            print(f"HELP TIMEOUT {s}")
            fail += 1

    # Quick sanity that DroFairTrainer history fields exist
    try:
        from src.training.dro_fair import DroFairTrainer
        from src.models.classifier import MLPClassifier
        import numpy as np
        import torch
        torch.manual_seed(0); np.random.seed(0)
        n, d = 100, 8
        X = np.random.randn(n, d).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        a = (X[:, 1] > 0).astype(np.int64)
        m = MLPClassifier(d, hidden_dims=[16], dropout=0.0)
        t = DroFairTrainer(m, alpha=0.1, device='cpu',
                           epochs=3, K_inner=2, tau_warmup_epochs=1)
        h = t.fit(X, y, a, verbose=False)
        keys = ['lambda_dp', 'lambda_if', 'g_dp', 'g_if']
        missing = [k for k in keys if k not in h or len(h[k]) != 3]
        if missing:
            print(f"HISTORY FAIL: missing/short {missing}")
            fail += 1
        else:
            print(f"HISTORY OK : {keys} present, len=3")
    except Exception as e:
        print(f"HISTORY FAIL: {e}")
        fail += 1

    if fail:
        print(f"\n{fail} check(s) failed")
        sys.exit(1)
    print("\nAll checks passed — safe to queue long jobs.")


if __name__ == '__main__':
    main()
