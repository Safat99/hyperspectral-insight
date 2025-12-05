import os
import json
import pandas as pd

BASE_DIR = "results/hyper3dnetlite/"

def summarize_lite_fullbands():
    rows = []

    # list dataset folders
    for dataset in os.listdir(BASE_DIR):
        dataset_dir = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        json_path = os.path.join(dataset_dir, f"{dataset}_lite_fullbands_cv.json")

        if not os.path.exists(json_path):
            print(f"[SKIP] No JSON found for {dataset}")
            continue

        print(f"Processing: {json_path}")

        # load JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        if "mean_metrics" not in data or "std_metrics" not in data:
            print(f"[WARNING] Missing metrics in {dataset}")
            continue

        row = {"dataset": dataset}

        # add all mean metrics
        for k, v in data["mean_metrics"].items():
            row[f"{k}_mean"] = v

        # add all std metrics
        for k, v in data["std_metrics"].items():
            row[f"{k}_std"] = v

        rows.append(row)

    # convert to DataFrame
    df = pd.DataFrame(rows)

    # sort alphabetically for readability
    df = df.sort_values("dataset")

    print("\n=== Full-Bands Summary Table ===\n")
    print(df)

    # output file
    out_path = os.path.join(BASE_DIR, "lite_fullbands_summary.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved summary CSV to:\n  {out_path}")


if __name__ == "__main__":
    summarize_lite_fullbands()
