# experiments/conv3d_full/run_3dcnn_srpa.py

import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.band_selection.srpa import run_srpa_pipeline, srpa_selection_3dcnn
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation
from hyperspectral_insight.models.attention_conv3d_full import build_full3dcnn


def run_3dcnn_srpa(
    dataset_name: str,
    num_bands: int = 20,
    patch_size: int = 11,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    save_dir: str = "results/conv3d_full/srpa/",
    verbose: bool = True,
    max_samples: int = 2000,
    lr: float = 5e-4,
):
    """
    Full 3D CNN with SRPA band selection.
    """

    print(f"\n=== 3D CNN + SRPA ({num_bands} bands) on {dataset_name} ===")

    # 1. Load
    cube, gt = load_dataset(dataset_name)

    # 2. Normalize
    cube_norm = minmax_normalize(cube)

    # 3. SRPA band selection
    print("Running SRPA band selection...")
    selected_bands, _ = srpa_selection_3dcnn(
        cube=cube_norm,
        gt=gt,
        num_bands=num_bands,
        batch_size=batch_size
    )

    print(f"  Selected bands ({num_bands}): {selected_bands}")

    # 4. Reduce cube
    cube_sel = cube_norm[:, :, selected_bands]

    # 5. Extract patches
    # X, y = extract_patches(cube_sel, gt, patch_size)
    X, y = extract_patches(
        cube_sel, gt,
        win=patch_size,
        drop_label0=True,
        max_samples_per_class=max_samples
    )
    print(f"  Patch shape: {X.shape}, #classes: {int(y.max()) + 1}")

    def model_fn(input_shape, n_classes):
        # If your build_hyper3dnet doesn't accept lr, remove lr=lr
        return build_full3dcnn(input_shape, n_classes, lr=lr)
    
    # 6. Stratified CV
    print("Running Stratified K-Fold CV...")
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

    # 7. Save
    os.makedirs(save_dir, exist_ok=True)
    out_json = os.path.join(
        save_dir, f"{dataset_name}_3dcnn_srpa_{num_bands}bands_{n_splits}fold_cv.json"
    )
    out_bands = os.path.join(
        save_dir, f"{dataset_name}_3dcnn_srpa_{num_bands}bands_{n_splits}_fold.cv.npy"
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    np.save(out_bands, np.array(selected_bands, dtype=np.int32))

    print(f"Saved CV results to: {out_json}")
    print(f"Saved band list to:  {out_bands}")

    return {
        "selected_bands": selected_bands,
        "cv_results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    run_3dcnn_srpa(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        lr=args.learning_rate,
        max_samples=args.max_samples,
    )
