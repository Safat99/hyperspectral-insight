import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_ssl_scaling_results(json_path, save_path="ssl_scaling_plot.png"):
    """
    Reads the json file created by run_ssl_scaling_experiment()
    and plots boxplots for:
        - baseline scores
        - ssl scores
    across subset fractions.
    """

    with open(json_path, "r") as f:
        results = json.load(f)

    # Sort subsets by size (just in case)
    subset_fracs = sorted([float(k) for k in results.keys()])

    baseline_data = []
    ssl_data = []
    labels = []

    for frac in subset_fracs:
        r = results[str(frac)]
        baseline_data.append(r["baseline_scores"])
        ssl_data.append(r["ssl_scores"])
        labels.append(f"{int(frac*100)}%")

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 6))

    # Create positions
    positions_baseline = np.arange(len(subset_fracs)) * 2.0
    positions_ssl = positions_baseline + 0.7

    # Baseline boxplots (red)
    plt.boxplot(
        baseline_data,
        positions=positions_baseline,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral"),
        medianprops=dict(color="darkred"),
    )

    # SSL boxplots (blue)
    plt.boxplot(
        ssl_data,
        positions=positions_ssl,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="navy"),
    )

    # Labels
    plt.xticks(
        positions_baseline + 0.35,
        labels,
        fontsize=12,
    )
    plt.ylabel("Overall Accuracy (OA)", fontsize=14)
    plt.xlabel("Subset Fraction Used for Training", fontsize=14)
    plt.title("Supervised Baseline vs SSL Across Dataset Sizes", fontsize=16)

    # Legend
    plt.plot([], [], color="lightcoral", label="Baseline (L only)", linewidth=10)
    plt.plot([], [], color="lightblue", label="SSL (L + U)", linewidth=10)
    plt.legend(fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True,
                        help="Path to *_ssl_scaling_results.json")
    parser.add_argument("--save", type=str, default="ssl_scaling_plot.png")

    args = parser.parse_args()

    plot_ssl_scaling_results(args.json, args.save)
