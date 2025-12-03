# Plots all 10 folds training curves

import numpy as np
from hyperspectral_insight.evaluation.visualization import plot_training_history


def analyze_histories(dataset="indian_pines"):
    hist_path = f"results/baseline/{dataset}_baseline_histories.npy"

    histories = np.load(hist_path, allow_pickle=True)

    print(f"Loaded {len(histories)} fold histories")

    for i, hist in enumerate(histories, 1):
        print(f"\n=== Plotting Fold {i} ===")
        plot_training_history(hist, title=f"{dataset} - Fold {i}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    args = parser.parse_args()

    analyze_histories(args.dataset)
