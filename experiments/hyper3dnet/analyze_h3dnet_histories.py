# experiments/hyper3dnet/analyze_h3dnet_histories.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for HPC / headless environments

from hyperspectral_insight.evaluation.visualization import (
    plot_training_history,
    plot_mean_std_history,
)


def analyze_histories(dataset: str = "indian_pines", variant: str = "original"):
    """
    Plot and save training curves for Hyper3DNet experiments (HPC-safe, always saves).

    variant:
        - "original" → uses {dataset}_h3dnet_original_histories.npy
        - "pca"      → uses {dataset}_h3dnet_pca_histories.npy
    """

    # Select correct file paths
    if variant == "original":
        hist_path = f"results/hyper3dnet/{dataset}_h3dnet_original_histories.npy"
        title_prefix = f"{dataset} - Hyper3DNet (Full bands)"
        save_prefix = f"results/hyper3dnet/{dataset}_h3dnet_original"
    elif variant == "pca":
        hist_path = f"results/hyper3dnet/{dataset}_h3dnet_pca_histories.npy"
        title_prefix = f"{dataset} - Hyper3DNet + PCA"
        save_prefix = f"results/hyper3dnet/{dataset}_h3dnet_pca"
    else:
        raise ValueError("variant must be 'original' or 'pca'")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"History file not found: {hist_path}")

    # Load histories
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
    parser.add_argument("--variant", type=str, default="original", choices=["original", "pca"])
    args = parser.parse_args()

    analyze_histories(dataset=args.dataset, variant=args.variant)
