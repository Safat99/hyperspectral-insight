# Plots all 10 folds training curves

import numpy as np
from hyperspectral_insight.evaluation.visualization import plot_training_history, plot_mean_std_history


def analyze_histories(dataset="indian_pines", save_plot=True):
    hist_path = f"results/baseline/{dataset}_baseline_histories.npy"
    
    histories = np.load(hist_path, allow_pickle=True)
    
    print(f"Loaded {len(histories)} fold histories")

    for i, hist in enumerate(histories, 1):
        print(f"\n=== Plotting Fold {i} ===")
        # Build a save path only if save_plot=True
        save_path = (
            f"results/baseline/{dataset}_Fold{i}.png"
            if save_plot else None
        )
        
        plot_training_history(
            history=hist,
            save_path=save_path,
            title=f"{dataset} - Fold {i}"
        )
    
    # === MEAN & STD PLOTS ===
    metrics = ["loss", "val_loss", "accuracy", "val_accuracy"]

    for metric in metrics:
        save_path = f"results/baseline/{dataset}_mean_{metric}.png"
        plot_mean_std_history(
            histories,
            metric=metric,
            title=f"{dataset} Mean ± Std {metric}",
            save_path=save_path
        )
        print(f"Saved mean/std plot: {save_path}")
        
        
    
    print("\nAll folds processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    args = parser.parse_args()

    analyze_histories(args.dataset)
