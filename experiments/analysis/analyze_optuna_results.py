import json
import pandas as pd
from pathlib import Path

# ------------------ CONFIG ------------------
RESULT_DIR = Path("results/hyper3dnetlite/new/optuna")
OUTPUT_DIR = RESULT_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
# --------------------------------------------


def load_all_trials(dataset_name: str) -> pd.DataFrame:
    """Load and merge all per-task CSVs for a dataset."""
    csv_files = sorted(RESULT_DIR.glob(f"{dataset_name}_trials_task*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No trial CSVs found for {dataset_name}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def load_best_json(dataset_name: str) -> dict:
    json_path = RESULT_DIR / f"{dataset_name}_optuna_best.json"
    with open(json_path) as f:
        return json.load(f)


def summarize_trials(df: pd.DataFrame) -> pd.Series:
    """Compute summary stats used in the paper."""
    completed = df[df["state"] == "COMPLETE"]

    return pd.Series({
        "n_trials_total": len(df),
        "n_trials_complete": len(completed),
        "best_f1": completed["value_f1"].max(),
        "mean_f1": completed["value_f1"].mean(),
        "std_f1": completed["value_f1"].std(),
        "median_f1": completed["value_f1"].median(),
        "mean_epochs": completed["trained_epochs"].mean(),
        "optimizer_adam_frac": (completed["optimizer"] == "adam").mean(),
        "optimizer_adadelta_frac": (completed["optimizer"] == "adadelta").mean(),
    })


def analyze_dataset(dataset_name: str):
    print(f"[INFO] Analyzing {dataset_name}")

    df = load_all_trials(dataset_name)
    best = load_best_json(dataset_name)

    # Save merged trial file
    merged_path = OUTPUT_DIR / f"{dataset_name}_all_trials.csv"
    df.to_csv(merged_path, index=False)

    # Summary stats
    summary = summarize_trials(df)
    summary["dataset"] = dataset_name
    summary["best_optimizer"] = best["best_params"]["optimizer"]
    summary["best_patch"] = best["best_params"]["patch_stride"]
    summary["best_batch"] = best["best_params"]["batch_size"]
    summary["best_oa"] = best["best_oa"]
    summary["best_kappa"] = best["best_kappa"]

    return summary


def main():
    datasets = sorted({
        p.name.replace("_optuna_best.json", "")
        for p in RESULT_DIR.glob("*_optuna_best.json")
    })

    summaries = []
    for ds in datasets:
        summaries.append(analyze_dataset(ds))

    summary_df = pd.DataFrame(summaries)

    # Save final tables
    summary_df.to_csv(OUTPUT_DIR / "summary_all_datasets.csv", index=False)
    summary_df.sort_values("best_f1", ascending=False).to_csv(
        OUTPUT_DIR / "summary_sorted_by_f1.csv", index=False
    )

    print("\n=== SUMMARY ===")
    print(summary_df[[
        "dataset",
        "best_f1",
        "best_oa",
        "best_optimizer",
        "mean_f1",
        "std_f1",
        "mean_epochs"
    ]])


if __name__ == "__main__":
    main()
