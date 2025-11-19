import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite


def run_lite_ibra_gss(
    dataset_name: str,
    num_bands: int = 20,
    vif_threshold: float = 10.0,
    max_distance: float = 5.0,
    patch_size: int = 11,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    save_dir: str = "results/hyper3dnetlite/",
    verbose: bool = True,
):
    """
    Hyper3DNet-Lite with IBRA+GSS band selection.

    Pipeline:
        1) Load dataset
        2) Normalize cube per band
        3) Run IBRA → GSS selection (unsupervised)
        4) Reduce cube to selected bands
        5) Extract patches
        6) Stratified K-Fold CV
        7) Save results (bands + performance)

    Args:
        dataset_name: dataset key (e.g., 'indian_pines')
        num_bands: number of final selected bands
        vif_threshold: VIF threshold for IBRA
        max_distance: IBRA pruning distance
        patch_size: patch window for 3D CNN
        n_splits: CV folds
        epochs: training epochs
        batch_size: training batch size
        save_dir: where to store CV metrics + selected bands
        verbose: print IBRA/GSS details
    """

    print(f"\n=== Hyper3DNet-Lite + IBRA+GSS ({num_bands} bands) on {dataset_name} ===")

    # -----------------------------------------
    # 1. Load dataset
    # -----------------------------------------
    cube, gt = load_dataset(dataset_name)

    # -----------------------------------------
    # 2. Normalize per band
    # -----------------------------------------
    cube_norm = minmax_normalize(cube)

    # -----------------------------------------
    # 3. Band-selection (IBRA → GSS)
    # -----------------------------------------
    print(f"Running IBRA+GSS band selection ...")

    selected_bands = select_bands_ibra_gss(
        cube=cube_norm,
        nbands=num_bands,
        vif_threshold=vif_threshold,
        max_distance=max_distance,
        verbose=verbose,
    )

    print(f"  Selected {num_bands} bands:")
    print(f"  {selected_bands}")

    # -----------------------------------------
    # 4. Reduce cube
    # -----------------------------------------
    cube_sel = cube_norm[:, :, selected_bands]

    # -----------------------------------------
    # 5. Patch extraction
    # -----------------------------------------
    X, y = extract_patches(cube_sel, gt, patch_size)

    print(f"  Patch shape: {X.shape}")
    print(f"  Number of classes: {int(y.max()) + 1}")

    # -----------------------------------------
    # 6. Stratified K-Fold CV
    # -----------------------------------------
    print("Running Stratified K-Fold CV...")

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

    # -----------------------------------------
    # 7. Save results
    # -----------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    out_json = os.path.join(
        save_dir,
        f"{dataset_name}_lite_ibra_gss_{num_bands}bands_cv.json",
    )
    out_bands = os.path.join(
        save_dir,
        f"{dataset_name}_lite_ibra_gss_{num_bands}bands.npy",
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    np.save(out_bands, np.array(selected_bands, dtype=np.int32))

    print(f"Saved CV results to: {out_json}")
    print(f"Saved selected bands to: {out_bands}")

    return {
        "selected_bands": selected_bands,
        "cv_results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--vif_threshold", type=float, default=10.0)
    parser.add_argument("--max_distance", type=float, default=5.0)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_lite_ibra_gss(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        vif_threshold=args.vif_threshold,
        max_distance=args.max_distance,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
