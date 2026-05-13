"""
Merge dataset-split experiment outputs into the standard results artifacts.

This is useful when the 150-run is executed as three parallel jobs:
  python3 experiments/run_experiments.py --datasets adult  --output_dir results/full_adult
  python3 experiments/run_experiments.py --datasets credit --output_dir results/full_credit
  python3 experiments/run_experiments.py --datasets lsac   --output_dir results/full_lsac
"""

import json
import os
import pickle

import numpy as np


DATASET_DIRS = {
    "adult": "results/full_adult",
    "credit": "results/full_credit",
    "lsac": "results/full_lsac",
}


def main():
    all_results = []
    runtime_rows = []

    for dataset, output_dir in DATASET_DIRS.items():
        path = os.path.join(output_dir, "all_results.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing completed results for {dataset}: {path}")

        with open(path) as f:
            rows = json.load(f)

        if len(rows) != 50:
            raise ValueError(f"{dataset} has {len(rows)} results; expected 50")

        bad_rows = [r for r in rows if r.get("dataset") != dataset]
        if bad_rows:
            raise ValueError(f"{dataset} output contains rows for another dataset")

        all_results.extend(rows)

        runtime_path = os.path.join(output_dir, "runtimes.json")
        if os.path.exists(runtime_path):
            with open(runtime_path) as f:
                runtime_rows.append(json.load(f))

    dataset_order = {"adult": 0, "credit": 1, "lsac": 2}
    all_results.sort(key=lambda r: (dataset_order[r["dataset"]], r["alpha"], r["seed"]))

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

    print(f"Merged {len(all_results)} results into results/all_results.json")
    print(f"Runtime overhead: {runtime_data['overhead']:.2f}x")


if __name__ == "__main__":
    main()
