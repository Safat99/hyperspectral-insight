import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.hyper3dnet import build_hyper3dnet
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation

def run_h3dnet_original_cv(
    dataset_name: str,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = "results/hyper3dnet/",
):
    """
    Full-band Hyper3DNet, NO PCA / band-selection.
    Stratified K-fold CV, ideal-paper style.

    Args:
        dataset_name: 'indian_pines', 'salinas', 'pavia', 'ksc', ...
        patch_size: spatial window (paper often uses 11)
        n_splits: # of folds for CV
        epochs: epochs per fold
        batch_size: batch size
        lr: learning rate for Hyper3DNet
        save_dir: where to store JSON results

    Saves:
        JSON file with per-fold metrics + mean/std metrics.
        NPY with list of history dicts (per fold)
    """

    print(f"\n=== Hyper3DNet ORIGINAL (no PCA/BS) on {dataset_name} ===")

    # 1) Load dataset
    cube, gt = load_dataset(dataset_name)

    # 2) Normalize per band
    cube_norm = minmax_normalize(cube)

    # 3) Extract patches → (N, H, W, B, 1), y → (N,)
    X, y = extract_patches(cube_norm, gt, patch_size)

    print(f"  Patches shape: {X.shape}, Labels shape: {y.shape}")
    print(f"  #classes (excluding 0): {int(y.max()) + 1}")
    
    def model_fn(input_shape, n_classes):
        # If your build_hyper3dnet doesn't accept lr, remove lr=lr
        return build_hyper3dnet(input_shape, n_classes, lr=lr)

    # 4) Run stratified K-fold cross-validation
    results = kfold_cross_validation(
        model_fn=model_fn,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        random_state=0,
        verbose=1,
    )

    # 5) Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_h3dnet_original_cv.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    # Histories are inside results["histories"] (list of dicts)
    out_hist = os.path.join(save_dir, f"{dataset_name}_h3dnet_original_histories.npy")
    np.save(out_hist, results["histories"], allow_pickle=True)

    print(f"  Saved metrics JSON to: {out_path}")
    print(f"  Saved fold histories to: {out_hist}")
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    run_h3dnet_original_cv(
        dataset_name=args.dataset,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )