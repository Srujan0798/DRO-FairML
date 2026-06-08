#!/usr/bin/env python3
"""
Aggregate ALL experimental results into a single analysis report.

Reads:
  - results/fairness_pgd_results.json       (tabular FairnessTargetedPGD)
  - results/utkface_results.json            (UTKFace baseline)
  - results/utkface_results_server.json     (UTKFace baseline, server)
  - results/utkface_lambda_max_cap.json     (item #2, H3)
  - results/utkface_alpha_sweep.json        (item #5, alpha {0.3,0.4})
  - results/utkface_fairness_pgd.json       (item #6, FPGD on images)
  - results/utkface_pixel_pgd.json          (item #3, H2)
  - results/utkface_randinit.json           (item #4, H1)
  - results/lambda_diagnostic.json          (item #1, tabular lambda traj)

Produces:
  - results/ALL_RESULTS_SUMMARY.md          (human-readable report)
  - results/utkface_all_results.json        (machine-readable aggregate)

Usage:
    venv/bin/python3 experiments/aggregate_all_results.py
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize_runs(runs, name):
    """Compute mean +/- std for naive vs dro across runs."""
    if not runs:
        return f"**{name}**: no data\n"

    # Handle both old format (nested clean/corrupted) and new format (flat)
    def extract_dp_acc(entry, method):
        m = entry.get(method, {})
        if 'clean' in m and 'corrupted' in m:
            # old format: evaluate on corrupted test
            return m['corrupted']['dp_violation'], m['corrupted']['accuracy']
        else:
            # new format: flat
            return m.get('dp_violation', np.nan), m.get('accuracy', np.nan)

    naive_dp = []
    naive_acc = []
    dro_dp = []
    dro_acc = []
    for r in runs:
        ndp, nacc = extract_dp_acc(r, 'naive')
        ddp, dacc = extract_dp_acc(r, 'dro')
        naive_dp.append(ndp)
        naive_acc.append(nacc)
        dro_dp.append(ddp)
        dro_acc.append(dacc)

    lines = [f"### {name} ({len(runs)} runs)"]
    lines.append(f"| Method | DP violation | Accuracy |")
    lines.append(f"|--------|-------------|----------|")
    lines.append(f"| Naive  | {np.mean(naive_dp):.4f} +/- {np.std(naive_dp):.4f} | {np.mean(naive_acc):.3f} +/- {np.std(naive_acc):.3f} |")
    lines.append(f"| DRO    | {np.mean(dro_dp):.4f} +/- {np.std(dro_dp):.4f} | {np.mean(dro_acc):.3f} +/- {np.std(dro_acc):.3f} |")

    # Delta
    dp_delta = np.mean(dro_dp) - np.mean(naive_dp)
    acc_delta = np.mean(dro_acc) - np.mean(naive_acc)
    lines.append(f"| Delta  | {dp_delta:+.4f} | {acc_delta:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def group_by(runs, key):
    out = {}
    for r in runs:
        out.setdefault(r.get(key), []).append(r)
    return out


def summarize_by_alpha(runs, name):
    """Summarize runs grouped by alpha."""
    if not runs:
        return f"**{name}**: no data\n"

    def extract_dp_acc(entry, method):
        m = entry.get(method, {})
        if 'clean' in m and 'corrupted' in m:
            return m['corrupted']['dp_violation'], m['corrupted']['accuracy']
        else:
            return m.get('dp_violation', np.nan), m.get('accuracy', np.nan)

    lines = [f"### {name} by alpha"]
    lines.append(f"| alpha | n | Naive DP | DRO DP | Delta DP | Naive Acc | DRO Acc |")
    lines.append(f"|-------|---|----------|--------|----------|-----------|---------|")

    for alpha, group in sorted(group_by(runs, 'alpha').items()):
        if alpha is None:
            continue
        naive_dp = []
        dro_dp = []
        naive_acc = []
        dro_acc = []
        for r in group:
            ndp, nacc = extract_dp_acc(r, 'naive')
            ddp, dacc = extract_dp_acc(r, 'dro')
            naive_dp.append(ndp)
            naive_acc.append(nacc)
            dro_dp.append(ddp)
            dro_acc.append(dacc)
        dp_delta = np.mean(dro_dp) - np.mean(naive_dp)
        lines.append(f"| {alpha} | {len(group)} | "
                     f"{np.mean(naive_dp):.4f} | {np.mean(dro_dp):.4f} | {dp_delta:+.4f} | "
                     f"{np.mean(naive_acc):.3f} | {np.mean(dro_acc):.3f} |")
    lines.append("")
    return "\n".join(lines)


def summarize_lambda_diagnostic(runs):
    if not runs:
        return "**Lambda diagnostic**: no data\n"
    lines = ["### Lambda trajectory diagnostic (tabular)"]
    lines.append(f"| config | n | final lambda_dp | final test DP |")
    lines.append(f"|--------|---|-----------------|---------------|")
    for tag, group in sorted(group_by(runs, 'tag').items()):
        lam_final = [r['history']['lambda_dp'][-1] for r in group]
        dp_final = [r['final']['dp'] for r in group]
        lines.append(f"| {tag} | {len(group)} | "
                     f"{np.mean(lam_final):.4f} +/- {np.std(lam_final):.4f} | "
                     f"{np.mean(dp_final):.4f} +/- {np.std(dp_final):.4f} |")
    lines.append("")
    return "\n".join(lines)


def summarize_lambda_max_cap(runs):
    """Special summary for H3 test: compare lambda_max=1.5 vs 0.5."""
    if not runs:
        return ""
    lines = ["### H3 test: lambda_max cap on UTKFace"]
    lines.append(f"| lambda_max | alpha | n | DRO DP | DRO Acc |")
    lines.append(f"|------------|-------|---|--------|---------|")
    for lmax, group in sorted(group_by(runs, 'lambda_max').items()):
        for alpha, agroup in sorted(group_by(group, 'alpha').items()):
            dp_vals = [r['dro']['dp_violation'] for r in agroup]
            acc_vals = [r['dro']['accuracy'] for r in agroup]
            lines.append(f"| {lmax} | {alpha} | {len(agroup)} | "
                         f"{np.mean(dp_vals):.4f} +/- {np.std(dp_vals):.4f} | "
                         f"{np.mean(acc_vals):.3f} +/- {np.std(acc_vals):.3f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    results_dir = Path('results')

    data = {
        'fairness_pgd': load_json(results_dir / 'fairness_pgd_results.json'),
        'utkface_baseline': load_json(results_dir / 'utkface_results.json'),
        'utkface_baseline_server': load_json(results_dir / 'utkface_results_server.json'),
        'utkface_lambda_max_cap': load_json(results_dir / 'utkface_lambda_max_cap.json'),
        'utkface_alpha_sweep': load_json(results_dir / 'utkface_alpha_sweep.json'),
        'utkface_fairness_pgd': load_json(results_dir / 'utkface_fairness_pgd.json'),
        'utkface_pixel_pgd': load_json(results_dir / 'utkface_pixel_pgd.json'),
        'utkface_randinit': load_json(results_dir / 'utkface_randinit.json'),
        'lambda_diagnostic': load_json(results_dir / 'lambda_diagnostic.json'),
    }

    sections = []
    sections.append("# DRO-FairML — Complete Results Summary\n")
    sections.append(f"Generated automatically.\n")

    # Tabular Fairness-PGD
    if data['fairness_pgd']:
        sections.append("## Tabular FairnessTargetedPGD\n")
        for ds in ['adult', 'credit', 'lsac']:
            ds_runs = [r for r in data['fairness_pgd'] if r.get('dataset') == ds]
            if ds_runs:
                sections.append(summarize_by_alpha(ds_runs, ds.capitalize()))

    # UTKFace baseline
    sections.append("## UTKFace Baseline\n")
    if data['utkface_baseline']:
        sections.append(summarize_by_alpha(data['utkface_baseline'], "UTKFace (local)"))
    if data['utkface_baseline_server']:
        sections.append(summarize_by_alpha(data['utkface_baseline_server'], "UTKFace (server)"))

    # H3: lambda_max cap
    if data['utkface_lambda_max_cap']:
        sections.append("## UTKFace H3 Diagnostic\n")
        sections.append(summarize_lambda_max_cap(data['utkface_lambda_max_cap']))

    # Alpha sweep
    if data['utkface_alpha_sweep']:
        sections.append("## UTKFace Alpha Sweep {0.3, 0.4}\n")
        sections.append(summarize_by_alpha(data['utkface_alpha_sweep'], "UTKFace alpha sweep"))

    # Fairness PGD on images
    if data['utkface_fairness_pgd']:
        sections.append("## UTKFace FairnessTargetedPGD\n")
        for attack in ['dp', 'if', 'combined']:
            attack_runs = [r for r in data['utkface_fairness_pgd'] if r.get('attack') == attack]
            if attack_runs:
                sections.append(summarize_by_alpha(attack_runs, f"Attack={attack}"))

    # H2: pixel PGD
    if data['utkface_pixel_pgd']:
        sections.append("## UTKFace Pixel-Space PGD (H2)\n")
        sections.append(summarize_by_alpha(data['utkface_pixel_pgd'], "Pixel PGD"))

    # H1: random init
    if data['utkface_randinit']:
        sections.append("## UTKFace Random-Init ResNet18 (H1)\n")
        sections.append(summarize_by_alpha(data['utkface_randinit'], "Random init"))

    # Lambda diagnostic
    if data['lambda_diagnostic']:
        sections.append("## Tabular Lambda Trajectory\n")
        sections.append(summarize_lambda_diagnostic(data['lambda_diagnostic']))

    # Write report
    report = "\n".join(sections)
    report_path = results_dir / 'ALL_RESULTS_SUMMARY.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report -> {report_path}")

    # Write aggregate JSON
    agg_path = results_dir / 'utkface_all_results.json'
    # Only include UTKFace keys
    utkface_agg = {k: v for k, v in data.items() if k.startswith('utkface') or k == 'lambda_diagnostic'}
    with open(agg_path, 'w') as f:
        json.dump(utkface_agg, f, indent=2)
    print(f"Aggregate JSON -> {agg_path}")


if __name__ == '__main__':
    main()
