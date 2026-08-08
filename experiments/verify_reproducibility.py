#!/usr/bin/env python3
"""TASK F — diff the re-run against the locked canonical.

Reads results/canonical_tau1_cosine.json and results/canonical_tau1.json,
pairs rows by (dataset, alpha, seed, method, attack), and reports:
- accuracy: should be byte-identical (or within 1e-10)
- DP: shift ~1e-7, no conclusion should move
- IF: old was noise (~1e-11), new is a real value

Produces results/reproducibility_diff.md.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    new = json.load(open("results/canonical_tau1_cosine.json"))
    old = json.load(open("results/canonical_tau1.json"))[:540]  # only the locked 540

    def key(r):
        return (r["dataset"], r["alpha"], r["seed"], r["method"], r["attack"])

    new_by = {key(r): r for r in new}
    old_by = {key(r): r for r in old}

    common = sorted(set(new_by) & set(old_by))
    print(f"Paired rows: {len(common)}")

    acc_diffs, dp_diffs, if_diffs = [], [], []
    rows = []
    for k in common:
        n, o = new_by[k], old_by[k]
        da = abs(n["acc_clean"] - o["acc_clean"])
        dd = abs(n["dp_clean"] - o["dp_clean"])
        di = abs(n["if_clean"] - o["if_clean"])
        acc_diffs.append(da)
        dp_diffs.append(dd)
        if_diffs.append(di)
        rows.append({
            "dataset": k[0], "alpha": k[1], "seed": k[2],
            "method": k[3], "attack": k[4],
            "acc_diff": da, "dp_diff": dd, "if_diff": di,
            "if_old": o["if_clean"], "if_new": n["if_clean"],
        })

    acc_diffs = np.array(acc_diffs)
    dp_diffs = np.array(dp_diffs)
    if_diffs = np.array(if_diffs)

    print(f"accuracy: max_diff={acc_diffs.max():.2e} mean={acc_diffs.mean():.2e} identical={np.sum(acc_diffs < 1e-10)}/{len(acc_diffs)}")
    print(f"DP:       max_diff={dp_diffs.max():.2e} mean={dp_diffs.mean():.2e}")
    print(f"IF:       max_diff={if_diffs.max():.2e} mean={if_diffs.mean():.2e}")

    # Check: old IF was noise (< 1e-9), new IF is real (> 1e-3)
    old_noise = sum(1 for r in rows if abs(r["if_old"]) < 1e-9)
    new_real = sum(1 for r in rows if abs(r["if_new"]) > 1e-3)
    print(f"Old IF was noise (< 1e-9): {old_noise}/{len(rows)}")
    print(f"New IF is real (> 1e-3): {new_real}/{len(rows)}")

    # Write markdown report
    lines = ["# TASK F — Canonical Reproducibility Diff\n",
             f"Paired rows: **{len(common)}**\n",
             f"| Metric | max | mean |",
             f"|---|---|---|",
             f"| accuracy | {acc_diffs.max():.2e} | {acc_diffs.mean():.2e} |",
             f"| DP | {dp_diffs.max():.2e} | {dp_diffs.mean():.2e} |",
             f"| IF | {if_diffs.max():.2e} | {if_diffs.mean():.2e} |",
             "",
             f"- Old IF was noise (< 1e-9): **{old_noise}/{len(rows)}**",
             f"- New IF is real (> 1e-3): **{new_real}/{len(rows)}**",
             "",
             "## Per-row detail\n",
             "| dataset | α | seed | attack | method | acc_diff | dp_diff | if_diff | if_old | if_new |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["dataset"], x["attack"], x["alpha"], x["seed"], x["method"])):
        lines.append(f"| {r['dataset']} | {r['alpha']} | {r['seed']} | {r['attack']} | {r['method']} | {r['acc_diff']:.2e} | {r['dp_diff']:.2e} | {r['if_diff']:.2e} | {r['if_old']:.2e} | {r['if_new']:.2e} |")

    os.makedirs("results", exist_ok=True)
    with open("results/reproducibility_diff.md", "w") as f:
        f.write("\n".join(lines))
    print("Wrote results/reproducibility_diff.md")

    # Verdict
    acc_ok = acc_diffs.max() < 1e-6
    dp_ok = dp_diffs.max() < 1e-4
    if_fixed = (old_noise > len(rows) * 0.5) and (new_real > len(rows) * 0.5)
    print(f"\nVerdict: accuracy_reproduces={acc_ok} DP_stable={dp_ok} IF_fixed={if_fixed}")
    if acc_ok and dp_ok:
        print("PASS — canonical reproducibility confirmed, DP story unaffected")
    else:
        print("INVESTIGATE — unexpected drift")


if __name__ == "__main__":
    main()
