
import os
import json
import pandas as pd

BASE_DIR = "results/baseline/"

def summarize_results():
    rows = []

    for fname in os.listdir(BASE_DIR):
        if not fname.endswith(".json"):
            continue

        dataset = fname.replace("_baseline.json", "")
        path = os.path.join(BASE_DIR, fname)

        with open(path, "r") as f:
            data = json.load(f)

        row = {"dataset": dataset}
        row.update(data["mean_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== Summary of Shallow Baselines ===\n")
    print(df)

    out_path = os.path.join(BASE_DIR, "baseline_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    summarize_results()