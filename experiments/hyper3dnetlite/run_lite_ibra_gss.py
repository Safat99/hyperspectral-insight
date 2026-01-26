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
    num_bands: int = 5,
    vif_threshold: float = 10.0,
    max_distance: float = 5.0,
    patch_size: int = 17,
    n_splits: int = 10,
    epochs: int = 25,
    batch_size: int = 32,
    optimizer: str = "adam",
    lr: float = 1e-3,
    rho: float = 0.95,
    epsilon: float = 1e-7,
    save_dir: str = "results/hyper3dnetlite/reduced_ibra_gss/",
    verbose: bool = True,
    max_samples_per_class: int = None, 
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
    print(f"Optimizer: {optimizer}")
    
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
    # X, y = extract_patches(cube_sel, gt, patch_size)
    X, y = extract_patches(
        cube_sel, gt,
        win=patch_size,
        drop_label0=True,
        max_samples_per_class=max_samples_per_class
    )

    print(f"  Patch shape: {X.shape}")
    print(f"  Number of classes: {int(y.max()) + 1}")

    # -----------------------------------------
    # 6. Stratified K-Fold CV
    # -----------------------------------------
    print("Running Stratified K-Fold CV...")

    def model_fn(input_shape, n_classes):
        if optimizer.lower() == "adam":
            return build_hyper3dnet_lite(
                input_shape,
                n_classes,
                optimizer_name="adam",
                lr=lr,
            )
        elif optimizer.lower() == "adadelta":
            return build_hyper3dnet_lite(
                input_shape,
                n_classes,
                optimizer_name="adadelta",
                rho=rho,
                epsilon=epsilon,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

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
        use_early_stopping=False,
    )

    # -----------------------------------------
    # 7. Save results
    # -----------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    out_json = os.path.join(
        save_dir,
        f"{dataset_name}_lite_ibra_gss_{num_bands}bands_{n_splits}fold_cv.json",
    )
    out_bands = os.path.join(
        save_dir,
        f"{dataset_name}_lite_ibra_gss_{num_bands}bands_{n_splits}fold_cv.npy",
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
    parser.add_argument("--patch", type=int, default=17)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

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
        optimizer=args.optimizer,
        lr=args.learning_rate,
        rho=args.rho,
        epsilon=args.epsilon,
        verbose=args.verbose,
        max_samples_per_class=args.max_samples,
    )
