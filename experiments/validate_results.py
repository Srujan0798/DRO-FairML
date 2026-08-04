#!/usr/bin/env python3
"""
Validation gate on the canonical tau=1 grid.

Checks that DRO-FAIR beats Naive-FAIR on DP under Wilcoxon p<0.05
for headline (dataset, alpha) cells using the DP attack.

Source of truth: results/canonical_tau1.json via load_canonical_tau1().
Never reads results/all_results.json nested legacy schema or stale_archived.
"""
import sys
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate():
    from experiments.loaders import load_canonical_tau1

    rows = load_canonical_tau1()
    print(f"Loaded canonical_tau1.json: {len(rows)} flat rows")

    # Prefer DP attack for the gate (paper headline). Fall back to any attack
    # present only if DP is completely absent (should not happen on full grid).
    attacks = sorted({r.get('attack') for r in rows})
    attack = 'dp' if 'dp' in attacks else (attacks[0] if attacks else None)
    if attack is None:
        print("ERROR: no attack field in canonical rows")
        return False
    print(f"Gate attack: {attack} (available: {attacks})")

    sub_all = [r for r in rows if r.get('attack') == attack]
    print(f"Rows for attack={attack}: {len(sub_all)}")

    wins = {"dp": 0, "if": 0, "total": 0}
    mean_wins = {"dp": 0, "if": 0}
    acc_drops = []

    print("\n" + "=" * 70)
    print("WILCOXON VALIDATION (one-sided H1: Naive DP > DRO DP, paired by seed)")
    print("Source: results/canonical_tau1.json (no all_results / stale fallback)")
    print("=" * 70)

    # Canonical alphas include 0.0; gate historically used 0.1/0.2/0.3.
    gate_alphas = [0.1, 0.2, 0.3]
    for ds in ['adult', 'credit', 'lsac']:
        for a in gate_alphas:
            cell = [
                r for r in sub_all
                if r['dataset'] == ds and abs(float(r['alpha']) - a) < 1e-6
            ]
            if not cell:
                print(f"{ds} a={a}: NO DATA")
                continue

            by_seed = defaultdict(dict)
            for r in cell:
                by_seed[int(r['seed'])][r['method']] = r

            paired = [
                (v['naive'], v['dro'])
                for v in by_seed.values()
                if 'naive' in v and 'dro' in v
            ]
            if len(paired) < 2:
                print(f"{ds} a={a}: insufficient paired seeds ({len(paired)})")
                continue

            n_dp = np.array([p[0]['dp_clean'] for p in paired], dtype=float)
            d_dp = np.array([p[1]['dp_clean'] for p in paired], dtype=float)
            n_if = np.array([p[0]['if_clean'] for p in paired], dtype=float)
            d_if = np.array([p[1]['if_clean'] for p in paired], dtype=float)
            n_acc = float(np.mean([p[0]['acc_clean'] for p in paired]))
            d_acc = float(np.mean([p[1]['acc_clean'] for p in paired]))

            diff_dp = n_dp - d_dp
            diff_if = n_if - d_if
            try:
                _, p_dp = wilcoxon(diff_dp, alternative='greater', zero_method='wilcox')
            except Exception:
                p_dp = 1.0
            try:
                _, p_if = wilcoxon(diff_if, alternative='greater', zero_method='wilcox')
            except Exception:
                p_if = 1.0

            dp_sig = (p_dp < 0.05) and (np.mean(d_dp) < np.mean(n_dp))
            if_sig = (p_if < 0.05) and (np.mean(d_if) < np.mean(n_if))
            dp_mean_win = np.mean(d_dp) < np.mean(n_dp)
            if_mean_win = np.mean(d_if) < np.mean(n_if)

            if dp_sig:
                wins["dp"] += 1
            if if_sig:
                wins["if"] += 1
            if dp_mean_win:
                mean_wins["dp"] += 1
            if if_mean_win:
                mean_wins["if"] += 1
            wins["total"] += 1
            acc_drops.append(n_acc - d_acc)

            dp_red = (
                (np.mean(n_dp) - np.mean(d_dp)) / np.mean(n_dp) * 100
                if np.mean(n_dp) > 0 else 0
            )
            if_red = (
                (np.mean(n_if) - np.mean(d_if)) / np.mean(n_if) * 100
                if np.mean(n_if) > 0 else 0
            )

            dp_status = (
                f"SIG_WIN(p={p_dp:.3f})" if dp_sig
                else (f"mean_win(p={p_dp:.3f})" if dp_mean_win else f"NOT_SIG(p={p_dp:.3f})")
            )
            if_status = (
                f"SIG_WIN(p={p_if:.3f})" if if_sig
                else (f"mean_win(p={p_if:.3f})" if if_mean_win else f"NOT_SIG(p={p_if:.3f})")
            )

            print(
                f"{ds:6s} a={a}: DP {np.mean(n_dp):.4f}->{np.mean(d_dp):.4f} "
                f"({dp_red:+.1f}%) {dp_status:20s} | "
                f"IF {np.mean(n_if):.4f}->{np.mean(d_if):.4f} "
                f"({if_red:+.1f}%) {if_status:20s} | "
                f"Acc {n_acc:.4f}->{d_acc:.4f} ({(n_acc - d_acc) * 100:.1f}% drop) "
                f"n={len(paired)}"
            )

    # Credit α=0.4 accuracy sanity (if present)
    credit_04 = [
        r for r in sub_all
        if r['dataset'] == 'credit' and abs(float(r['alpha']) - 0.4) < 1e-6
        and r['method'] == 'dro'
    ]
    if credit_04:
        d_acc04 = float(np.mean([r['acc_clean'] for r in credit_04]))
        print(f"\nCredit a=0.4 DRO acc: {d_acc04:.4f} "
              f"{'OK' if d_acc04 >= 0.60 else 'WARNING: low'}")

    print(f"\n{'=' * 70}")
    print(f"DP WINS (Wilcoxon p<0.05):  {wins['dp']}/{wins['total']}  (need >= 6/9)")
    print(f"IF WINS (Wilcoxon p<0.05):  {wins['if']}/{wins['total']}  (report claims 5/9)")
    print(f"DP WINS (mean-based only):  {mean_wins['dp']}/{wins['total']}  (for reference)")
    print(f"IF WINS (mean-based only):  {mean_wins['if']}/{wins['total']}  (for reference)")
    if acc_drops:
        print(f"Avg accuracy drop: {np.mean(acc_drops) * 100:.2f}%")
    print(f"{'=' * 70}")

    if wins['total'] == 0:
        print("RESULT: FAIL (no gate cells found in canonical_tau1.json)")
        print(f"{'=' * 70}")
        return False

    passed = wins['dp'] >= 6
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")
    return passed


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
