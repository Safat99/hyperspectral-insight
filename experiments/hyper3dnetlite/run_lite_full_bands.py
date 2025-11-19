import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def run_lite_fullbands_cv(
    dataset_name: str,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 128,
    save_dir: str = "results/hyper3dnetlite/",
):
    """
    Hyper3DNet-Lite full-band cross-validation baseline.
    No PCA, no band-selection.
    Stratified K-fold CV with OA/F1/Kappa.

    Args:
        dataset_name: e.g. 'indian_pines'
        patch_size: spatial window for 3D patches (paper uses 25)
        n_splits: number of CV folds (default 10)
        epochs: training epochs per fold
        batch_size: training batch size
        save_dir: directory to save results JSON

    Returns:
        results: dict containing per-fold metrics and mean/std metrics
    """

    print(f"\n=== Hyper3DNet-Lite FULL-BANDS CV on {dataset_name} ===")

    # --------------------------
    # 1. Load dataset
    # --------------------------
    cube, gt = load_dataset(dataset_name)

    # --------------------------
    # 2. Normalize cube per band
    # --------------------------
    cube_norm = minmax_normalize(cube)

    # --------------------------
    # 3. Extract 3D patches
    # --------------------------
    X, y = extract_patches(cube_norm, gt, patch_size)
    print(f"  Patches: {X.shape}, Labels: {y.shape}")
    print(f"  Classes: {int(y.max()) + 1}")

    # --------------------------
    # 4. Stratified K-Fold CV
    # --------------------------
    print("  Running Stratified K-Fold CV...")

    results = kfold_cross_validation(
        model_fn=build_hyper3dnet_lite,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        random_state=0,
        verbose=0,
    )

    # --------------------------
    # 5. Save results JSON
    # --------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_lite_fullbands_cv.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"  Saved results to: {out_path}")
    print(f"  Mean metrics: {results['mean_metrics']}")
    print(f"  Std metrics:  {results['std_metrics']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    run_lite_fullbands_cv(
        dataset_name=args.dataset,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )