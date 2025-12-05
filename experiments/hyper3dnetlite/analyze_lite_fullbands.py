import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe for HPC jobs, avoids opening windows

from hyperspectral_insight.evaluation.visualization import (
    plot_training_history,
    plot_mean_std_history,
)

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
BASE_DIR = "results/hyper3dnetlite/"
HIST_SUFFIX = "_h3dnetlite_fullbands_histories.npy"
JSON_SUFFIX = "_lite_fullbands_cv.json"


# ------------------------------------------------------------------------
# SINGLE DATASET ANALYSIS
# ------------------------------------------------------------------------
def analyze_dataset(dataset_name: str, dataset_dir: str):
    """Process .npy fold histories + generate per-fold & mean/std plots."""

    hist_path = os.path.join(dataset_dir, f"{dataset_name}{HIST_SUFFIX}")
    json_path = os.path.join(dataset_dir, f"{dataset_name}{JSON_SUFFIX}")

    if not os.path.exists(hist_path):
        print(f"[SKIP] No histories found for {dataset_name}")
        return

    print(f"\n=== Processing dataset: {dataset_name} ===")
    print(f"→ Histories: {hist_path}")

    # Load histories
    histories = np.load(hist_path, allow_pickle=True)
    print(f"Loaded {len(histories)} folds")

    save_prefix = os.path.join(dataset_dir, f"{dataset_name}_h3dnetlite_fullbands")
    title_prefix = f"{dataset_name} – Hyper3DNet-Lite (Full bands)"

    # ------------------------------------------------------------
    # PER-FOLD PLOTS
    # ------------------------------------------------------------
    for i, hist in enumerate(histories, 1):
        print(f"  Plotting fold {i} ...")
        outfile = f"{save_prefix}_Fold{i}.png"
        plot_training_history(
            history=hist,
            title=f"{title_prefix} – Fold {i}",
            save_path=outfile
        )
        print(f"    Saved: {outfile}")

    # ------------------------------------------------------------
    # MEAN / STD PLOTS
    # ------------------------------------------------------------
    metrics = ["loss", "val_loss", "accuracy", "val_accuracy"]

    for metric in metrics:
        outfile = f"{save_prefix}_mean_{metric}.png"
        print(f"  Plotting Mean±Std for {metric} ...")

        plot_mean_std_history(
            histories,
            metric=metric,
            title=f"{title_prefix} – Mean ± Std of {metric}",
            save_path=outfile
        )
        print(f"    Saved: {outfile}")

    print(f"✓ Completed dataset: {dataset_name}")


# ------------------------------------------------------------------------
# MASTER RUNNER – detect all datasets automatically
# ------------------------------------------------------------------------
def analyze_all():
    print(f"\n=== Auto-Scanning {BASE_DIR} ===")

    if not os.path.isdir(BASE_DIR):
        raise RuntimeError(f"Base directory not found: {BASE_DIR}")

    dataset_dirs = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    if not dataset_dirs:
        print("[ERROR] No dataset folders found.")
        return

    print(f"Found: {dataset_dirs}")

    for dataset in dataset_dirs:
        dataset_path = os.path.join(BASE_DIR, dataset)
        analyze_dataset(dataset_name=dataset, dataset_dir=dataset_path)

    print("\n=== All datasets processed successfully ===")


# ------------------------------------------------------------------------
if __name__ == "__main__":
    analyze_all()
