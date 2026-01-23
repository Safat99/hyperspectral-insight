import os
import json
import itertools
import numpy as np
import pandas as pd

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize

from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def tune_lite(
    # dataset_name: str = "indian_pines",
    # save_dir: str = "results/hyper3dnetlite/new/",
    # n_splits: int = 3,
    # epochs: int = 10,
    dataset_name: str,
    save_dir: str,
    config_id: int,
    n_splits: int = 5,
    epochs: int = 10,
    phase: int = 1,
):
    """
    Two-phase tuning for Hyper3DNet-Lite.
    Phase 1: screen (epochs=3)
    Phase 2: confirm (epochs=10)
    """

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== TUNING Hyper3DNet-Lite | Dataset={dataset_name} | Config={config_id} | Phase={phase} ===")

    # 1) Load dataset
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # Hyperparameter grid
    patch_stride_pairs = [
        (5, 1),
        (25, 1),
        (50, 25),
    ]
    batch_sizes = [16, 64, 128]
    optimizers = ["adam", "adadelta"]

    configs = list(itertools.product(patch_stride_pairs, batch_sizes, optimizers))
    
    if config_id >= len(configs):
        raise ValueError("Invalid config_id")
    
    (patch_size, stride), batch_size, opt_name = configs[config_id]

    print(f"Config details → patch={patch_size}, stride={stride}, batch={batch_size}, opt={opt_name}")
    
    # ---------------- Phase settings ----------------
    if phase == 1:
        run_epochs = 3
        max_samples = 300
    else:
        run_epochs = epochs
        max_samples = 1000
        
    # ---------------- Patch cache ----------------
    patch_cache = {}
    cache_key = (patch_size, stride)

    if cache_key not in patch_cache:
        X, y = extract_patches(
            cube_norm,
            gt,
            win=patch_size,
            stride=stride,
            drop_label0=True,
            max_samples_per_class=max_samples,
        )
        patch_cache[cache_key] = (X, y)
    else:
        X, y = patch_cache[cache_key]

    print(f"Patch tensor: X={X.shape}, y={y.shape}")

    # ---------------- Model factory ----------------
    def model_fn(input_shape, n_classes):
        if opt_name == "adadelta":
            return build_hyper3dnet_lite(
                input_shape, n_classes, optimizer_name="adadelta"
            )
        else:
            return build_hyper3dnet_lite(
                input_shape, n_classes, optimizer_name="adam", lr=1e-3
            )


    # ---------------- Cross-validation ----------------
    results = kfold_cross_validation(
        model_fn=model_fn,
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=run_epochs,
        batch_size=batch_size,
        shuffle=True,
        random_state=0,
        verbose=1,
        max_samples_per_class=max_samples,
    )

    row = {
        "dataset": dataset_name,
        "patch_size": patch_size,
        "stride": stride,
        "batch_size": batch_size,
        "optimizer": opt_name,
        "phase": phase,
        "epochs": run_epochs,
        "cv_folds": n_splits,
        **results["mean_metrics"],
    }
    
    out_path = os.path.join(
        save_dir, f"{dataset_name}_cfg{config_id}_phase{phase}.json"
    )

    with open(out_path, "w") as f:
        json.dump(row, f, indent=4)

    print(f"Saved results → {out_path}")
    print(f"Mean OA = {row['oa']:.4f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="results/h3dnetlite/tuning")

    args = parser.parse_args()

    tune_lite(
        dataset_name=args.dataset,
        save_dir=args.save_dir,
        config_id=args.config_id,
        n_splits=args.splits,
        epochs=args.epochs,
        phase=args.phase,
    )
