import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import create_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.shallow_cnn import build_shallow_cnn
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation

def run_shallow_baseline(
    dataset_name: str,
    patch_size: int = 25,
    n_splits: int = 10,
    save_dir: str = "results/baseline/"
):
    """
    Run 10-fold CV using shallow CNN on a specified dataset.
    """

    print(f"\n=== Running Shallow CNN Baseline on {dataset_name} ===")

    # 1) Load dataset (cube, gt)
    cube, gt = load_dataset(dataset_name)

    # 2) Normalize per band
    cube_norm = minmax_normalize(cube)

    # 3) Extract patches
    print("Extracting patches...")
    X, y = create_patches(cube_norm, gt, patch_size)

    # 4) Cross-validation
    print("Running cross-validation...")
    results = kfold_cross_validation(
        model_fn=build_shallow_cnn,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    # 5) Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_baseline.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved baseline results to: {out_path}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=25)
    args = parser.parse_args()

    run_shallow_baseline(args.dataset, patch_size=args.patch)
