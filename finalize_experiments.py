#!/usr/bin/env python3
"""Run this after lambda grid or canonical finishes to refresh all derived artifacts."""
import subprocess, os, json, time, sys

def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    $ {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("ERROR:", res.stderr[-500:] if res.stderr else res.stdout[-500:])
        return False
    print(res.stdout[-800:] if len(res.stdout) > 800 else res.stdout)
    return True

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    print("=== FINALIZE: waiting/refresh cycle ===")
    # Lambda
    try:
        lam = json.load(open("results/lambda_lr_grid.json"))
        print(f"lambda rows now: {len(lam)}/72")
    except:
        pass

    # Canonical
    try:
        can = json.load(open("results/canonical_tau1.json"))
        print(f"canonical rows now: {len(can)}/540")
    except:
        pass

    if len(lam) >= 72:
        print("Lambda grid COMPLETE -> refreshing summaries and figures")
        run("python3 experiments/analyze_tau1.py", "analyze_tau1 (figures + tau1_summary + wilcoxon)")
        run("python3 experiments/generate_report_tables.py", "report tables")
        run("(cd report && /opt/homebrew/bin/tectonic report.tex)", "report pdf")
        run("(cd paper && /opt/homebrew/bin/tectonic main.tex)", "paper pdf")
        print("Done for lambda. Commit the updated summary + pdfs + new figs.")
    else:
        print("Lambda not yet 72. Run this again later or let it finish.")

    # Always safe to refresh canonical-derived if more rows
    if len(can) > 60:   # we had ~69 before
        print("Canonical has progressed -> you can re-run analyze if you want Credit numbers, but full 540 recommended first.")
    
    print("\nWhen BOTH are done: re-run this script, then git add results/tau1_* figures/ report/ paper/ and commit 'FINAL data + tables'.")

if __name__ == "__main__":
    main()
