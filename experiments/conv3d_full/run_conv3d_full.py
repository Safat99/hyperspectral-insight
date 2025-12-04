import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.models.attention_conv3d_full import build_full3dcnn
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def run_conv3d_full(
    dataset_name: str,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    save_dir: str = "results/conv3d_full/",
    verbose: bool = True,
):
    """
    FULL-BAND Conv3D CNN (no band selection).
    Saves:
        - k-fold training history
        - mean metrics
        - cv results as json
    """

    print(f"\n=== FULL-BAND 3D CNN on {dataset_name} ===")

    # 1. Load dataset
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # 2. Extract patches
    X, y = extract_patches(cube_norm, gt, patch_size)
    print(f"  Patch shape: {X.shape}, Classes: {int(y.max()) + 1}")

    # 3. Run k-fold CV
    print("Running Stratified K-Fold CV...")
    
    results = kfold_cross_validation(
        model_fn=build_full3dcnn,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        random_state=0,
        verbose=0,
    )

    # 4. Save results
    os.makedirs(save_dir, exist_ok=True)

    out_json = os.path.join(
        save_dir,
        f"{dataset_name}_3dcnn_fullbands_patch{patch_size}_cv.json"
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved CV results: {out_json}")
    print("Training history stored per fold.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=11)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_conv3d_full(
        dataset_name=args.dataset,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
