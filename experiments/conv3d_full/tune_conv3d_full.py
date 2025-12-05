import os
import json
import itertools
import numpy as np
import pandas as pd

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.attention_conv3d_full import build_full3dcnn
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def tune_conv3d_full(
    dataset_name: str = "indian_pines",
    save_dir: str = "results/conv3d_full_tuning/",
    n_splits: int = 3,
    epochs: int = 10,
    max_samples_per_class: int = None,
):
    """
    Lightweight hyperparameter tuning for FULL-BAND Conv3D CNN.

    Grid:
        patch_size ∈ {11, 25}
        batch_size ∈ {8, 16, 32}
        lr ∈ {1e-3, 5e-4, 1e-4}
    """

    print(f"\n=== TUNING Full-Band Conv3D CNN on {dataset_name} ===")

    # Load full cube
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # Hyperparameter grid
    patch_sizes = [25]
    batch_sizes = [4, 16, 32, 64]
    lrs = [1e-3, 5e-4, 1e-4]

    configs = list(itertools.product(patch_sizes, batch_sizes, lrs))

    rows = []

    for patch_size, batch_size, lr in configs:
        print("\n----------------------------------------")
        print(f"Config: patch={patch_size}, batch={batch_size}, lr={lr}")

        # X, y = extract_patches(cube_norm, gt, patch_size)
        X, y = extract_patches(
        cube_norm, gt,
        win=patch_size,
        drop_label0=True,
        max_samples_per_class=max_samples_per_class
    )
        
        print(f"  Patches: {X.shape}")

        # Model builder with LR override
        def model_fn(input_shape, n_classes):
            return build_full3dcnn(input_shape, n_classes, lr=lr)

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
            lr=lr,  # <-- if your CV supports passing extra args
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
        }

        rows.append(row)
        print(f"  → Mean OA: {mean_oa:.4f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    json_path = os.path.join(save_dir, f"{dataset_name}_conv3d_tuning.json")
    csv_path = os.path.join(save_dir, f"{dataset_name}_conv3d_tuning.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)

    df.to_csv(csv_path, index=False)

    print("\n=== BEST CONFIG ===")
    print(df.iloc[df["mean_oa"].idxmax()])

    print(f"\nSaved tuning table:\n  {json_path}\n  {csv_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    tune_conv3d_full(
        dataset_name=args.dataset,
        n_splits=args.splits,
        epochs=args.epochs,
        max_samples_per_class=args.max_samples,
    )
