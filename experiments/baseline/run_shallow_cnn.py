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
    epochs: int = 50,
    batch_size: int = 32,
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
    print(f"  X: {X.shape}, y: {y.shape}, classes: {int(y.max()) + 1}")

    # 4) Cross-validation
    print("Running cross-validation...")
    
    results = kfold_cross_validation(
        model_fn=build_shallow_cnn,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # 5) Save results
    os.makedirs(save_dir, exist_ok=True)
    
    out_path = os.path.join(save_dir, f"{dataset_name}_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Histories (list of dicts)
    out_hist = os.path.join(save_dir, f"{dataset_name}_baseline_histories.npy")
    np.save(out_hist, results["histories"], allow_pickle=True)

    print(f"Saved baseline results to: {out_path}")
    print(f"Saved fold histories to: {out_hist}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()

    run_shallow_baseline(
        dataset_name=args.dataset, 
        patch_size=args.patch,
        epochs=args.epochs,
        batch_size=args.batch_sizes,
    )
