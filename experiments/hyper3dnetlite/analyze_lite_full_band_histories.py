import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe: prevents GUI usage

from hyperspectral_insight.evaluation.visualization import (
    plot_training_history,
    plot_mean_std_history,
)


def analyze_histories(dataset: str = "indian_pines"):
    """
    Plot and save training curves for Hyper3DNet-Lite FULL-BANDS CV.
    Always saves plots (HPC/headless environment).
    """

    # Paths based on run_lite_fullbands_cv() output
    hist_path = f"results/hyper3dnetlite/{dataset}_h3dnetlite_fullbands_histories.npy"
    save_prefix = f"results/hyper3dnetlite/{dataset}_h3dnetlite_fullbands"
    title_prefix = f"{dataset} - Hyper3DNet-Lite (Full bands)"

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"History file not found: {hist_path}")

    # Load fold histories
    histories = np.load(hist_path, allow_pickle=True)
    print(f"Loaded {len(histories)} fold histories from {hist_path}")

    # === PER-FOLD PLOTS ===
    for i, hist in enumerate(histories, 1):
        print(f"\n=== Plotting Fold {i} ===")

        save_path = f"{save_prefix}_Fold{i}.png"

        plot_training_history(
            history=hist,
            title=f"{title_prefix} - Fold {i}",
            save_path=save_path
        )

        print(f"Saved: {save_path}")

    # === MEAN & STD PLOTS ===
    metrics = ["loss", "val_loss", "accuracy", "val_accuracy"]

    for metric in metrics:
        save_path = f"{save_prefix}_mean_{metric}.png"

        plot_mean_std_history(
            histories,
            metric=metric,
            title=f"{title_prefix} Mean ± Std {metric}",
            save_path=save_path
        )

        print(f"Saved mean/std plot: {save_path}")

    print("\nAll folds processed and saved successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    args = parser.parse_args()

    analyze_histories(dataset=args.dataset)
