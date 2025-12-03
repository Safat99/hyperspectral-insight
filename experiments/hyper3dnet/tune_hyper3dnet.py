# experiments/hyper3dnet/tune_hyper3dnet.py

import os
import json
import itertools
import numpy as np
import pandas as pd

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.hyper3dnet import build_hyper3dnet
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def tune_hyper3dnet(
    dataset_name: str = "indian_pines",
    save_dir: str = "results/hyper3dnet/",
    n_splits: int = 3,
    epochs: int = 20,
):
    """
    Lightweight hyperparameter tuning for Hyper3DNet.

    Grid:
        patch_size ∈ {11, 25}
        batch_size ∈ {8, 16}
        lr ∈ {1e-3, 5e-4, 1e-4}
    """

    print(f"\n=== TUNING Hyper3DNet on {dataset_name} ===")

    # 1) Load dataset
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # Hyperparameter grid
    patch_sizes = [25]
    batch_sizes = [4, 8, 16]
    lrs = [1e-3, 5e-4, 1e-4]

    configs = list(itertools.product(patch_sizes, batch_sizes, lrs))

    rows = []

    for patch_size, batch_size, lr in configs:
        print("\n----------------------------------------")
        print(f"Config: patch={patch_size}, batch={batch_size}, lr={lr}")

        # Extract patches for this patch size
        X, y = extract_patches(cube_norm, gt, patch_size)
        print(f"  X: {X.shape}, y: {y.shape}")

        def model_fn(input_shape, n_classes):
            # If your build_hyper3dnet doesn't take lr, remove lr=lr
            return build_hyper3dnet(input_shape, n_classes, lr=lr)

        results = kfold_cross_validation(
            model_fn=model_fn,
            X=X,
            y=y,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            random_state=0,
            verbose=0,
        )

        mean_oa = results["mean_metrics"]["oa"]
        row = {
            "dataset": dataset_name,
            "patch_size": patch_size,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "cv_folds": n_splits,
            "mean_oa": mean_oa,
            "mean_precision": results["mean_metrics"]["precision"],
            "mean_recall": results["mean_metrics"]["recall"],
            "mean_f1": results["mean_metrics"]["f1"],
            "mean_kappa": results["mean_metrics"]["kappa"],
        }
        rows.append(row)

        print(f"  → mean OA: {mean_oa:.4f}")

    # ---------------- Save tuning results ----------------
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    json_path = os.path.join(save_dir, f"{dataset_name}_h3dnet_tuning.json")
    csv_path = os.path.join(save_dir, f"{dataset_name}_h3dnet_tuning.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)

    df.to_csv(csv_path, index=False)

    # Best config
    best_row = df.iloc[df["mean_oa"].idxmax()]
    print("\n=== BEST CONFIG ===")
    print(best_row)

    print(f"\nSaved tuning table to:\n  {json_path}\n  {csv_path}")

    return df, best_row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    tune_hyper3dnet(
        dataset_name=args.dataset,
        n_splits=args.splits,
        epochs=args.epochs,
    )
