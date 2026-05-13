"""
Merge dataset-split experiment outputs into the standard results artifacts.

Supports both:
  A) Single-process per dataset: results/full_adult, results/full_credit, results/full_lsac
  B) Parallel-by-alpha (Adult only): results/adult_a0.0, results/adult_a0.1, ..., results/adult_a0.4
"""

import json
import os
import pickle

import numpy as np


def load_results_from_dir(output_dir, expected_count=None):
    path = os.path.join(output_dir, "all_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f)
    return rows


def load_from_checkpoint(output_dir):
    path = os.path.join(output_dir, "checkpoint.pkl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get("results", [])
    except Exception:
        return []


def main():
    all_results = []
    runtime_rows = []

    # Credit and LSAC (single-process)
    for dataset in ["credit", "lsac"]:
        output_dir = f"results/full_{dataset}"
        rows = load_results_from_dir(output_dir)
        if not rows:
            rows = load_from_checkpoint(output_dir)
        if rows:
            all_results.extend(rows)
            print(f"  {dataset}: {len(rows)} results")
        else:
            print(f"  {dataset}: no data yet")

        runtime_path = os.path.join(output_dir, "runtimes.json")
        if os.path.exists(runtime_path):
            with open(runtime_path) as f:
                runtime_rows.append(json.load(f))

    # Adult (parallel-by-alpha or single-process)
    adult_rows = []
    # Try parallel structure first
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
        output_dir = f"results/adult_a{alpha}"
        rows = load_results_from_dir(output_dir)
        if not rows:
            rows = load_from_checkpoint(output_dir)
        if rows:
            adult_rows.extend(rows)
            print(f"  adult α={alpha}: {len(rows)} results")

    # Fallback to single-process structure
    if not adult_rows:
        rows = load_results_from_dir("results/full_adult")
        if not rows:
            rows = load_from_checkpoint("results/full_adult")
        if rows:
            adult_rows.extend(rows)
            print(f"  adult (single): {len(rows)} results")

    if adult_rows:
        all_results.extend(adult_rows)

    if not all_results:
        print("No results found yet.")
        return

    dataset_order = {"adult": 0, "credit": 1, "lsac": 2}
    all_results.sort(key=lambda r: (dataset_order.get(r["dataset"], 99), r["alpha"], r["seed"]))

    os.makedirs("results", exist_ok=True)
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("results/all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    naive_times = [r["naive"]["time"] for r in all_results]
    dro_times = [r["dro"]["time"] for r in all_results]
    runtime_data = {
        "naive_mean": float(np.mean(naive_times)) if naive_times else 0.0,
        "naive_std": float(np.std(naive_times)) if naive_times else 0.0,
        "dro_mean": float(np.mean(dro_times)) if dro_times else 0.0,
        "dro_std": float(np.std(dro_times)) if dro_times else 0.0,
        "overhead": float(np.mean(dro_times) / np.mean(naive_times)) if np.mean(naive_times) > 0 else 0.0,
        "n_experiments": len(all_results),
        "split_runtime_files": runtime_rows,
    }
    with open("results/runtimes.json", "w") as f:
        json.dump(runtime_data, f, indent=2)

    print(f"\nMerged {len(all_results)} results into results/all_results.json")
    print(f"Runtime overhead: {runtime_data['overhead']:.2f}x")


if __name__ == "__main__":
    main()
