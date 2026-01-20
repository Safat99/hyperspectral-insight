import os
import json
import numpy as np
import argparse

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import create_patches
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.models.shallow_cnn import build_shallow_cnn
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation

def run_tuning(dataset_name, patch_size=25):

    print(f"\n=== Tuning Shallow CNN on {dataset_name} ===")

    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    X, y = create_patches(cube_norm, gt, patch_size)
    print(f"  X: {X.shape}, y: {y.shape}, classes={int(y.max())+1}")

    # Tuning space (light)
    batch_sizes = [16, 32, 64, 128]
    lrs = [1e-3, 5e-4, 1e-4]

    results = []

    for bs in batch_sizes:
        for lr in lrs:
            print(f"\n--- Testing batch={bs}, lr={lr} ---")
            
            def model_fn(input_shape, n_classes):
                return build_shallow_cnn(input_shape, n_classes, lr=lr)

            cv = kfold_cross_validation(
                model_fn=model_fn,
                X=X,
                y=y,
                n_splits=3,         # faster tuning
                epochs=50,          # fewer epochs
                batch_size=bs,
                verbose=0,
                max_samples_per_class=2000
            )

            row = {
                "batch": bs,
                "lr": lr,
                "mean_OA": cv["mean_metrics"]["oa"],
                "mean_F1": cv["mean_metrics"]["f1"],
                "mean_Precision": cv["mean_metrics"]["precision"],
                "mean_Recall": cv["mean_metrics"]["recall"]
            }
            results.append(row)

    out_dir = "results/baseline/new/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_tuning_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved tuning results → {out_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    run_tuning(args.dataset)