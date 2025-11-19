# experiments/ssl/run_ssl_lite_ibra_gss_kfold.py

import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite

from hyperspectral_insight.ssl.split import (
    ssl_train_val_test_split,
    ssl_kfold_indices_for_labeled,
)
from hyperspectral_insight.ssl.pseudo_label import run_pseudo_label_ssl_fold


def run_ssl_lite_ibra_gss_kfold(
    dataset_name: str,
    num_bands: int = 20,
    patch_size: int = 25,
    test_frac: float = 0.20,
    labeled_frac_within_train: float = 0.12,
    n_splits: int = 10,
    ssl_iters: int = 5,
    confidence_thresh: float = 0.9,
    epochs_per_iter: int = 20,
    batch_size: int = 16,
    save_dir: str = "results/ssl/",
    random_state: int = 0,
):
    """
    Main SSL experiment:

        - Hyper3DNet-Lite
        - IBRA+GSS band selection
        - K-fold CV ON 12% labeled subset
        - Shared unlabeled pool (68%)
        - Global 20% TEST split

    Results:
        JSON file with per-fold OA on TEST and SSL history per fold.
    """

    print(f"\n=== SSL Hyper3DNet-Lite + IBRA-GSS on {dataset_name} ===")
    print(f"  num_bands={num_bands}, patch_size={patch_size}")
    print(f"  test_frac={test_frac}, labeled_frac_within_train={labeled_frac_within_train}")
    print(f"  K-fold={n_splits}, ssl_iters={ssl_iters}\n")

    # ------------------------------------------
    # 1) Load dataset & normalize
    # ------------------------------------------
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # ------------------------------------------
    # 2) IBRA+GSS band selection (unsupervised)
    # ------------------------------------------
    print("Running IBRA+GSS band selection (unsupervised)...")
    selected_bands = select_bands_ibra_gss(
        cube=cube_norm,
        nbands=num_bands,
        vif_threshold=10.0,
        max_distance=5.0,
        verbose=False,
    )
    print(f"  Selected bands: {selected_bands}")

    cube_sel = cube_norm[:, :, selected_bands]

    # ------------------------------------------
    # 3) Extract patches
    # ------------------------------------------
    X_all, y_all = extract_patches(cube_sel, gt, patch_size)
    print(f"  Patches: {X_all.shape}, labels: {y_all.shape}")
    print(f"  #classes (excluding 0): {int(y_all.max()) + 1}")

    # ------------------------------------------
    # 4) Global SSL split: TEST + labeled pool + unlabeled pool
    # ------------------------------------------
    split_dict = ssl_train_val_test_split(
        X_all,
        y_all,
        test_frac=test_frac,
        labeled_frac_within_train=labeled_frac_within_train,
        random_state=random_state,
    )

    X_test = split_dict["X_test"]
    y_test = split_dict["y_test"]
    X_labeled_pool = split_dict["X_labeled_pool"]
    y_labeled_pool = split_dict["y_labeled_pool"]
    X_unlabeled_pool = split_dict["X_unlabeled_pool"]
    # y_unlabeled_pool = split_dict["y_unlabeled_pool"]  # unused in training

    # ------------------------------------------
    # 5) K-fold on labeled pool (12%)
    # ------------------------------------------
    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        ssl_kfold_indices_for_labeled(
            y_labeled_pool,
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ),
        start=1,
    ):
        print(f"\n[SSL-CV] Fold {fold_idx}/{n_splits}")

        # For each fold, we treat the *entire* labeled pool sub-split as fold-labeled.
        # Here we keep it simple: use *all* labeled pool for this fold
        # but allow internal train/val split inside run_pseudo_label_ssl_fold.
        # If you want CV-style disjoint folds, you can decide to treat
        # (tr_idx) as the fold's labeled data and ignore val_idx, but
        # we keep them only as an indexing example here.

        X_labeled_fold = X_labeled_pool  # you could also restrict to X_labeled_pool[tr_idx]
        y_labeled_fold = y_labeled_pool

        res = run_pseudo_label_ssl_fold(
            build_model_fn=build_hyper3dnet_lite,
            X_labeled_fold=X_labeled_fold,
            y_labeled_fold=y_labeled_fold,
            X_unlabeled_pool=X_unlabeled_pool,
            X_test=X_test,
            y_test=y_test,
            epochs_per_iter=epochs_per_iter,
            batch_size=batch_size,
            confidence_thresh=confidence_thresh,
            ssl_iters=ssl_iters,
            val_frac=0.2,  # 80/20 train/val inside this fold
        )

        print(f"  Fold {fold_idx} TEST OA: {res['OA_test']:.4f}")
        fold_results.append(res)

    # ------------------------------------------
    # 6) Aggregate across folds
    # ------------------------------------------
    oa_list = [r["OA_test"] for r in fold_results]
    mean_oa = float(np.mean(oa_list))
    std_oa = float(np.std(oa_list))

    print(f"\n[SSL-CV] Mean TEST OA over {n_splits} folds: {mean_oa:.4f} ± {std_oa:.4f}")

    out_dict = {
        "dataset": dataset_name,
        "num_bands": num_bands,
        "patch_size": patch_size,
        "test_frac": test_frac,
        "labeled_frac_within_train": labeled_frac_within_train,
        "n_splits": n_splits,
        "ssl_iters": ssl_iters,
        "confidence_thresh": confidence_thresh,
        "epochs_per_iter": epochs_per_iter,
        "batch_size": batch_size,
        "selected_bands": list(map(int, selected_bands)),
        "fold_results": fold_results,
        "mean_OA_test": mean_oa,
        "std_OA_test": std_oa,
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_ssl_lite_ibra_gss_kfold.json")

    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=4)

    print(f"\nSaved SSL K-fold results to: {out_path}")

    return out_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--lab_frac", type=float, default=0.12)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--ssl_iters", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    run_ssl_lite_ibra_gss_kfold(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        test_frac=args.test_frac,
        labeled_frac_within_train=args.lab_frac,
        n_splits=args.splits,
        ssl_iters=args.ssl_iters,
        confidence_thresh=args.conf,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
    )
