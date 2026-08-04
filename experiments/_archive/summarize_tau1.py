#!/usr/bin/env python3
"""Summarize the tau=1 vs tau=10 vs tau=100 ablation for the report/paper.

Reads the JSON files produced by `experiments/run_tau_ablation.py` and emits
a Markdown table of (Adult, DP attack, Naive vs DRO) under each fixed tau
plus the k-NN ablation numbers. The output is the authoritative table the
report/paper cite.  We do NOT write to `results/` (Agent A owns that); we
just print + write to docs/.
"""
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU_FILES = {
    1: os.path.join(REPO, "results", "tau_ablation_tau1.json"),
    10: os.path.join(REPO, "results", "tau_ablation_tau10.json"),
    100: os.path.join(REPO, "results", "tau_ablation_tau100.json"),
}
# Optional override via env var so we can read a committed snapshot
# without racing an in-progress run that has truncated the live file.
import os as _os
_TAU1_OVERRIDE = _os.environ.get("TAU1_JSON_OVERRIDE")
if _TAU1_OVERRIDE:
    TAU_FILES[1] = _TAU1_OVERRIDE
KNN_FILES = {
    k: os.path.join(REPO, "results", f"knn_ablation_k{k}.json")
    for k in (5, 10, 15)
}
RAND_VS_ADV = os.path.join(REPO, "results", "random_vs_adversarial_new.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _mean(xs):
    return mean(xs) if xs else float("nan")


def _se(xs):
    return pstdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0


def tau_comparison_table():
    """Adult, DP attack, naive vs dro DP at tau ∈ {1, 10, 100}."""
    tau_data = {tau: _load(p) for tau, p in TAU_FILES.items() if os.path.exists(p)}
    rows = []
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4):
        for tau in (1, 10, 100):
            if tau not in tau_data:
                continue
            entry = {"alpha": alpha, "tau": tau}
            for method in ("naive", "dro"):
                runs = [
                    r for r in tau_data[tau]
                    if r["dataset"] == "adult"
                    and r["attack"] == "dp"
                    and r["alpha"] == alpha
                    and r["method"] == method
                ]
                dps = [r["dp_clean"] for r in runs]
                accs = [r["acc_clean"] for r in runs]
                entry[f"{method}_dp_mean"] = _mean(dps)
                entry[f"{method}_dp_se"] = _se(dps)
                entry[f"{method}_acc_mean"] = _mean(accs)
                entry[f"{method}_n"] = len(dps)
            # Per-seed win (DRO DP < Naive DP)
            wins = 0
            seeds_done = 0
            naive_by_seed = {
                r["seed"]: r for r in tau_data[tau]
                if r["dataset"] == "adult" and r["attack"] == "dp"
                and r["alpha"] == alpha and r["method"] == "naive"
            }
            dro_by_seed = {
                r["seed"]: r for r in tau_data[tau]
                if r["dataset"] == "adult" and r["attack"] == "dp"
                and r["alpha"] == alpha and r["method"] == "dro"
            }
            for s in set(naive_by_seed) & set(dro_by_seed):
                seeds_done += 1
                if dro_by_seed[s]["dp_clean"] < naive_by_seed[s]["dp_clean"]:
                    wins += 1
            entry["dro_wins"] = wins
            entry["seeds_done"] = seeds_done
            rows.append(entry)
    return rows, tau_data


def adult_tau1_summary():
    """The headline table for the report: Adult tau=1, all 3 attacks."""
    tau_data = _load(TAU_FILES[1])
    out = {}
    for attack in ("dp", "combined", "if"):
        out[attack] = []
        for alpha in (0.0, 0.1, 0.2, 0.3, 0.4):
            row = {"alpha": alpha}
            for method in ("naive", "dro"):
                runs = [
                    r for r in tau_data
                    if r["dataset"] == "adult"
                    and r["attack"] == attack
                    and r["alpha"] == alpha
                    and r["method"] == method
                ]
                dps = [r["dp_clean"] for r in runs]
                accs = [r["acc_clean"] for r in runs]
                row[f"{method}_dp"] = _mean(dps)
                row[f"{method}_acc"] = _mean(accs)
                row[f"{method}_n"] = len(dps)
            # wins
            naive_by_seed = {
                r["seed"]: r for r in tau_data
                if r["dataset"] == "adult" and r["attack"] == attack
                and r["alpha"] == alpha and r["method"] == "naive"
            }
            dro_by_seed = {
                r["seed"]: r for r in tau_data
                if r["dataset"] == "adult" and r["attack"] == attack
                and r["alpha"] == alpha and r["method"] == "dro"
            }
            wins = sum(
                1 for s in set(naive_by_seed) & set(dro_by_seed)
                if dro_by_seed[s]["dp_clean"] < naive_by_seed[s]["dp_clean"]
            )
            row["dro_wins"] = wins
            row["seeds_done"] = len(set(naive_by_seed) & set(dro_by_seed))
            out[attack].append(row)
    return out


def knn_table():
    """Adult, IF attack, k ∈ {5, 10, 15}."""
    out = []
    for k, path in KNN_FILES.items():
        if not os.path.exists(path):
            continue
        data = _load(path)
        for alpha in (0.1, 0.2, 0.3):
            row = {"k": k, "alpha": alpha}
            for method in ("naive", "dro"):
                runs = [
                    r for r in data
                    if r.get("alpha") == alpha and r.get("method") == method
                ]
                if not runs:
                    continue
                row[f"{method}_if"] = _mean([r["if_clean"] for r in runs])
                row[f"{method}_dp"] = _mean([r["dp_clean"] for r in runs])
            out.append(row)
    return out


def random_vs_adv_table():
    """The attack-vs-noise comparison (Adult/Credit/LSAC, α ∈ {0.1, 0.2, 0.3})."""
    if not os.path.exists(RAND_VS_ADV):
        return []
    data = _load(RAND_VS_ADV)
    out = []
    for r in data:
        dp_clean = r["clean"]["dp"]
        dp_rand = r["random"]["dp"]
        dp_adv = r["adversarial"]["dp"]
        delta_rand = dp_rand - dp_clean
        delta_adv = dp_adv - dp_clean
        ratio = delta_adv / delta_rand if delta_rand > 0.001 else float("nan")
        out.append({
            "dataset": r["dataset"],
            "alpha": r["alpha"],
            "clean_dp": dp_clean,
            "rand_delta": delta_rand,
            "adv_delta": delta_adv,
            "adv_over_rand": ratio,
        })
    return out


def print_markdown():
    print("# Tau=1 ablation summary")
    print()
    print("Source: `results/tau_ablation_tau{1,10,100}.json` (auto-generated by")
    print("`experiments/run_tau_ablation.py` with the same hyperparameters as the")
    print("main run, only `get_temperature` monkey-patched to return a constant).")
    print()

    print("## Adult, DP attack, fixed-tau comparison (mean ± SE over seeds done)")
    print()
    rows, _ = tau_comparison_table()
    print("| α | τ | Naive DP | DRO DP | Δacc (DRO−Naive) | DRO wins | seeds |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        dp_n = r["naive_dp_mean"]
        dp_d = r["dro_dp_mean"]
        da = r["dro_acc_mean"] - r["naive_acc_mean"]
        print(
            f"| {r['alpha']:.1f} | {r['tau']} | {dp_n:.4f}±{r['naive_dp_se']:.4f} | "
            f"{dp_d:.4f}±{r['dro_dp_se']:.4f} | {da:+.4f} | "
            f"{r['dro_wins']}/{r['seeds_done']} | {r['seeds_done']} |"
        )
    print()

    print("## Adult, tau=1, all three attacks (Naive vs DRO DP)")
    print()
    s = adult_tau1_summary()
    for attack in ("dp", "combined", "if"):
        print(f"### Attack = {attack.upper()}")
        print("| α | Naive DP | DRO DP | Δacc (DRO−Naive) | DRO wins | seeds |")
        print("|---|---|---|---|---|---|")
        for r in s[attack]:
            da = r["dro_acc"] - r["naive_acc"]
            print(
                f"| {r['alpha']:.1f} | {r['naive_dp']:.4f} | {r['dro_dp']:.4f} | "
                f"{da:+.4f} | {r['dro_wins']}/{r['seeds_done']} | "
                f"{r['seeds_done']} |"
            )
        print()

    print("## Adult, IF attack, k-NN ablation (mean IF / DP at each k)")
    print()
    print("Source: `results/knn_ablation_k{5,10,15}.json` (Adult only, 3 seeds).")
    print()
    print("| k | α | Naive IF | DRO IF | Naive DP | DRO DP |")
    print("|---|---|---|---|---|---|")
    for r in knn_table():
        print(
            f"| {r['k']} | {r['alpha']:.1f} | {r.get('naive_if',float('nan')):.4f} | "
            f"{r.get('dro_if',float('nan')):.4f} | "
            f"{r.get('naive_dp',float('nan')):.4f} | "
            f"{r.get('dro_dp',float('nan')):.4f} |"
        )
    print()

    print("## Adversarial vs random corruption (Δ DP from clean baseline)")
    print()
    print("Source: `results/random_vs_adversarial_new.json` (3 seeds per cell).")
    print()
    print("| Dataset | α | Adv ΔDP | Random ΔDP | Adv/Random ratio |")
    print("|---|---|---|---|---|")
    for r in random_vs_adv_table():
        ratio = r["adv_over_rand"]
        ratio_s = f"{ratio:.1f}×" if ratio == ratio else "n/a (rand≈0)"
        print(
            f"| {r['dataset']} | {r['alpha']:.1f} | {r['adv_delta']:+.4f} | "
            f"{r['rand_delta']:+.4f} | {ratio_s} |"
        )
    print()


if __name__ == "__main__":
    print_markdown()
