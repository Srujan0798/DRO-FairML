#!/usr/bin/env python3
"""Generate Accuracy vs alpha plots broken down by attack type (Kuldeep request)."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.plot_tau1_headline import clean_axes, savefig, COLORS, METHOD_LABEL

with open("results/canonical_tau1.json") as f:
    rows = json.load(f)

adult = [r for r in rows if r['dataset'] == 'adult']

for attack in ['dp', 'if', 'combined']:
    sub = [r for r in adult if r['attack'] == attack]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for meth, marker in [("naive", "o"), ("dro", "s")]:
        recs = []
        for alpha in sorted(set(r['alpha'] for r in sub)):
            vals = [r['acc_clean'] for r in sub if r['method'] == meth and r['alpha'] == alpha]
            vals = np.array(vals)
            if len(vals):
                recs.append({'alpha': alpha, 'mean': np.mean(vals), 'se': np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0})
        df = pd.DataFrame(recs).sort_values('alpha')
        if not df.empty:
            ax.errorbar(df['alpha'], df['mean'], yerr=df['se'],
                        marker=marker, color=COLORS[meth], label=METHOD_LABEL[meth],
                        linewidth=1.6, capsize=3, markersize=5.5)
    clean_axes(ax)
    ax.set_xlabel(r"corruption $\alpha$")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Adult Accuracy vs $\\alpha$ — {attack.upper()} attack ($\\tau = 1$)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xticks(sorted(set(r['alpha'] for r in sub)))
    # no y-limit — show all data including alpha=0.3 (which drops below 0.78)
    fig.tight_layout()
    stem = f"adult_acc_{attack}_attack_tau1_meeting"
    savefig(fig, stem)

print("Done. Saved adult_acc_{dp,if,combined}_attack_tau1_meeting.{pdf,png}")
