import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from hyperspectral_insight.evaluation.visualization import plot_training_history

def plot_all_histories(dataset="indian_pines", baseline_dir="results/baseline/"):

    hist_path = os.path.join(baseline_dir, f"{dataset}_baseline_histories.npy")
    histories = np.load(hist_path, allow_pickle=True)

    # --- 1) Individual folds ---
    out_dir = os.path.join(baseline_dir, f"{dataset}_plots/")
    os.makedirs(out_dir, exist_ok=True)

    for i, h in enumerate(histories):
        plt.figure(figsize=(10,4))
        plot_training_history(h, title=f"Fold {i+1} Training History")
        plt.savefig(os.path.join(out_dir, f"fold_{i+1}.png"))
        plt.close()

    # --- 2) Mean curves ---
    max_len = max(len(h["loss"]) for h in histories)

    # Pad histories (unequal lengths)
    losses = np.array([np.pad(h["loss"], (0, max_len - len(h["loss"])), 'edge')
                       for h in histories])
    val_losses = np.array([np.pad(h["val_loss"], (0, max_len - len(h["val_loss"])), 'edge')
                           for h in histories])
    acc = np.array([np.pad(h["accuracy"], (0, max_len - len(h["accuracy"])), 'edge')
                    for h in histories])
    val_acc = np.array([np.pad(h["val_accuracy"], (0, max_len - len(h["val_accuracy"])), 'edge')
                        for h in histories])

    # Compute means & std
    mean_loss = losses.mean(axis=0)
    std_loss = losses.std(axis=0)
    mean_val_loss = val_losses.mean(axis=0)
    std_val_loss = val_losses.std(axis=0)

    mean_acc = acc.mean(axis=0)
    std_acc = acc.std(axis=0)
    mean_val_acc = val_acc.mean(axis=0)
    std_val_acc = val_acc.std(axis=0)

    # --- Plot mean curves ---
    epochs = range(max_len)
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, mean_loss, label="mean loss")
    plt.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss, alpha=0.3)
    plt.plot(epochs, mean_val_loss, label="mean val loss")
    plt.fill_between(epochs, mean_val_loss-std_val_loss, mean_val_loss+std_val_loss, alpha=0.3)
    plt.title("Mean Loss Across Folds")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, mean_acc, label="mean acc")
    plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, alpha=0.3)
    plt.plot(epochs, mean_val_acc, label="mean val acc")
    plt.fill_between(epochs, mean_val_acc-std_val_acc, mean_val_acc+std_val_acc, alpha=0.3)
    plt.title("Mean Accuracy Across Folds")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_training_curves.png"))
    plt.close()

    print(f"Saved plots → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    plot_all_histories(args.dataset)
