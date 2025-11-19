# experiments/band_selection/run_ssep.py

import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.band_selection.ssep import run_ssep_pipeline
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite


def run_ssep(
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
    SSEP band selection + Hyper3DNet-Lite CV evaluation.

    Steps:
        1) Load dataset
        2) Normalize
        3) SSEP → top-K bands (supervised)
        4) Reduce cube
        5) Patch extraction
        6) Stratified 10-fold CV with Hyper3DNet-Lite
        7) Save results

    Args:
        dataset_name: e.g., 'indian_pines'
        num_bands: number of final selected bands
        patch_size: spatial patch size
        n_splits: CV folds
        epochs: training epochs
        batch_size: training batch
        save_dir: results directory
    """

    print(f"\n=== SSEP ({num_bands} bands) on {dataset_name} ===")

    # Load
    cube, gt = load_dataset(dataset_name)

    # Normalize
    cube_norm = minmax_normalize(cube)

    # SSEP band-selection
    print("Running SSEP band selection...")
    selected_bands = run_ssep_pipeline(
        cube=cube_norm,
        gt=gt,
        nbands=num_bands,
        verbose=verbose,
    )

    print(f"  Selected bands ({num_bands}): {selected_bands}")

    # Reduce cube
    cube_sel = cube_norm[:, :, selected_bands]

    # Extract patches
    X, y = extract_patches(cube_sel, gt, patch_size)
    print(f"  Patch shape: {X.shape}")
    print(f"  #classes: {y.max()+1}")

    # CV evaluation
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
    out_json = os.path.join(save_dir, f"{dataset_name}_ssep_{num_bands}bands_hyper3dnetlite_cv.json")
    out_bands = os.path.join(save_dir, f"{dataset_name}_ssep_{num_bands}bands_hyper3dnetlite.npy")

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

    run_ssep(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
