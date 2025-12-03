# experiments/hyper3dnet/analyze_h3dnet_histories.py

import os
import numpy as np

from hyperspectral_insight.evaluation.visualization import plot_training_history


def analyze_histories(dataset: str = "indian_pines", variant: str = "original"):
    """
    Plot training curves for Hyper3DNet experiments.

    variant:
        - "original" → uses {dataset}_h3dnet_original_histories.npy
        - "pca"      → uses {dataset}_h3dnet_pca_histories.npy
    """

    if variant == "original":
        hist_path = f"results/hyper3dnet/{dataset}_h3dnet_original_histories.npy"
        title_prefix = f"{dataset} - Hyper3DNet (Full bands)"
    elif variant == "pca":
        hist_path = f"results/hyper3dnet/{dataset}_h3dnet_pca_histories.npy"
        title_prefix = f"{dataset} - Hyper3DNet + PCA"
    else:
        raise ValueError("variant must be 'original' or 'pca'")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"History file not found: {hist_path}")

    histories = np.load(hist_path, allow_pickle=True)
    print(f"Loaded {len(histories)} fold histories from {hist_path}")

    for i, hist in enumerate(histories, 1):
        print(f"\n=== Plotting Fold {i} ===")
        plot_training_history(hist, title=f"{title_prefix} - Fold {i}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--variant", type=str, default="original", choices=["original", "pca"])
    args = parser.parse_args()

    analyze_histories(dataset=args.dataset, variant=args.variant)
