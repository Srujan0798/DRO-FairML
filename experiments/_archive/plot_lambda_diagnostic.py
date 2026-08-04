#!/usr/bin/env python3
"""
Plot lambda_dp trajectory diagnostic.

Reads results/lambda_diagnostic.json (output of run_lambda_diagnostic.py) and
produces a 2-panel figure:
  Left  : lambda_dp(epoch) curves, mean +/- std across seeds, one line per tag.
  Right : final test DP-violation bar chart per config.

If H3 holds, failing configs (adult lmax=1.5) should show monotone runaway
to lambda_max; capping (adult lmax=0.5) should bound it and reduce final DP.

Usage:
    venv/bin/python3 experiments/plot_lambda_diagnostic.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path='results/lambda_diagnostic.json'):
    with open(path) as f:
        return json.load(f)


def by_tag(runs):
    out = {}
    for r in runs:
        out.setdefault(r['tag'], []).append(r)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results/lambda_diagnostic.json')
    args = ap.parse_args()
    runs = load(args.results)
    groups = by_tag(runs)
    tags = list(groups.keys())

    colors = {
        'adult_lmax1.5':  '#d62728',  # red — known failure
        'adult_lmax0.5':  '#ff9f1c',  # orange — intervention
        'credit_lmax1.5': '#2ca02c',  # green — known success
        'lsac_lmax1.5':   '#1f77b4',  # blue — known success
    }
    labels = {
        'adult_lmax1.5':  'Adult  (lambda_max=1.5)  — known failure',
        'adult_lmax0.5':  'Adult  (lambda_max=0.5)  — intervention',
        'credit_lmax1.5': 'Credit (lambda_max=1.5)  — known success',
        'lsac_lmax1.5':   'LSAC   (lambda_max=1.5)  — known success',
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={'width_ratios': [1.6, 1.0]})

    final_dp = {}
    for tag in tags:
        runs_t = groups[tag]
        traj = np.array([r['history']['lambda_dp'] for r in runs_t])  # (seeds, epochs)
        mean = traj.mean(axis=0)
        std = traj.std(axis=0)
        ep = np.arange(1, traj.shape[1] + 1)
        c = colors.get(tag, '#888')
        axL.plot(ep, mean, color=c, linewidth=2, label=labels.get(tag, tag))
        axL.fill_between(ep, mean - std, mean + std, color=c, alpha=0.15)
        final_dp[tag] = [r['final']['dp'] for r in runs_t]

    axL.set_xlabel('Epoch')
    axL.set_ylabel(r'$\lambda_{DP}$')
    axL.set_title(r'$\lambda_{DP}$ trajectory  (alpha=0.2, DP attack, 3 seeds)')
    axL.legend(loc='best', fontsize=9)
    axL.grid(alpha=0.3)

    # Right panel: final test DP violation
    ordered = ['adult_lmax1.5', 'adult_lmax0.5', 'credit_lmax1.5', 'lsac_lmax1.5']
    ordered = [t for t in ordered if t in final_dp]
    means = [np.mean(final_dp[t]) for t in ordered]
    stds  = [np.std(final_dp[t])  for t in ordered]
    cs    = [colors.get(t, '#888') for t in ordered]
    xs = np.arange(len(ordered))
    axR.bar(xs, means, yerr=stds, color=cs, alpha=0.85, capsize=4)
    axR.set_xticks(xs)
    axR.set_xticklabels([t.replace('_lmax', '\nlmax=') for t in ordered],
                        fontsize=9)
    axR.set_ylabel('Test DP violation')
    axR.set_title('Final test DP (lower = fairer)')
    axR.grid(axis='y', alpha=0.3)

    plt.suptitle(r'Lambda trajectory diagnostic — H3: inner-max amplifies on continuous embeddings',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_pdf = 'figures/fig11_lambda_diagnostic.pdf'
    out_png = 'figures/fig11_lambda_diagnostic.png'
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f"Saved -> {out_pdf}\nSaved -> {out_png}")

    # Print summary table
    print("\n=== Summary (mean +/- std across seeds) ===")
    print(f"{'config':<22} {'final lambda_dp':>18} {'final test DP':>16}")
    for tag in ordered:
        rt = groups[tag]
        lam_final = np.array([r['history']['lambda_dp'][-1] for r in rt])
        dp_final  = np.array([r['final']['dp'] for r in rt])
        print(f"{tag:<22} {lam_final.mean():>10.3f} +/- {lam_final.std():<5.3f}  "
              f"{dp_final.mean():>8.4f} +/- {dp_final.std():.4f}")


if __name__ == '__main__':
    main()
