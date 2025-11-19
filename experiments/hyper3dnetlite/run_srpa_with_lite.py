# experiments/band_selection/run_srpa.py

import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.band_selection.srpa import run_srpa_pipeline
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite


def run_srpa(
    dataset_name: str,
    num_bands: int = 20,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 128,
    save_dir: str = "results/hyper3dnetlite/",
    verbose: bool = True,
):
    """
    SRPA (RF-Based Spectral Redundancy + Pixel Attention)
    PLUS Hyper3DNet-Lite evaluation.

    Steps:
        1. Load dataset
        2. Normalize per band
        3. SRPA → top-K supervised bands
        4. Reduce cube to selected bands
        5. Extract patches
        6. Stratified 10-fold CV
        7. Save output
    """

    print(f"\n=== SRPA ({num_bands} bands) on {dataset_name} ===")

    # Load
    cube, gt = load_dataset(dataset_name)

    # Normalize
    cube_norm = minmax_normalize(cube)

    # SRPA band-selection
    print("Running SRPA band selection...")
    selected_bands = run_srpa_pipeline(
        cube=cube_norm,
        gt=gt,
        nbands=num_bands,
        verbose=verbose,
    )

    print(f"  Selected bands ({num_bands}): {selected_bands}")

    # Reduce HSI
    cube_sel = cube_norm[:, :, selected_bands]

    # Patches
    X, y = extract_patches(cube_sel, gt, patch_size)
    print(f"  Patch shape: {X.shape}")

    # Cross validation
    print("Running Stratified CV...")
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

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_json = os.path.join(save_dir, f"{dataset_name}_lite_srpa_{num_bands}bands_cv.json")
    out_bands = os.path.join(save_dir, f"{dataset_name}_lite_srpa_{num_bands}bands.npy")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    np.save(out_bands, np.array(selected_bands, dtype=np.int32))

    print(f"Saved CV results: {out_json}")
    print(f"Saved band list:  {out_bands}")

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
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_srpa(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
