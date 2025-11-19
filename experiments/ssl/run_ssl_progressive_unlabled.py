# # experiments/ssl/run_ssl_progressive_unlabeled.py

# import os
# import json
# import numpy as np

# from hyperspectral_insight.data.loaders import load_dataset
# from hyperspectral_insight.data.patches import extract_patches
# from hyperspectral_insight.data.normalization import minmax_normalize

# from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
# from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite

# from hyperspectral_insight.ssl.split import split_ssl_full
# from hyperspectral_insight.ssl.pseudo_label import run_pseudo_label_ssl


# def run_ssl_progressive_unlabeled(
#     dataset_name: str,
#     patch_size: int = 25,
#     num_bands: int = 20,
#     test_frac: float = 0.20,
#     labeled_frac: float = 0.05,
#     val_frac: float = 0.10,
#     unlabeled_fracs=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85),
#     epochs: int = 20,
#     batch_size: int = 32,
#     ssl_iters: int = 5,
#     confidence_thresh: float = 0.9,
#     vif_threshold: float = 10.0,
#     max_distance: float = 5.0,
#     save_dir: str = "results/ssl/",
# ):
#     """
#     Run SSL multiple times with increasing unlabeled pool size.

#     - Labeled set and test set are fixed
#     - Unlabeled set increases: fraction of the *unlabeled pool* is used

#     This produces a curve: OA vs unlabeled fraction.
#     """

#     print(f"\n=== SSL Progressive Unlabeled ({dataset_name}) ===")
#     print(f"  Labeled_frac={labeled_frac}, test_frac={test_frac}, val_frac={val_frac}")
#     print(f"  Unlabeled fractions: {unlabeled_fracs}")

#     # 1) Load dataset
#     cube, gt = load_dataset(dataset_name)

#     # 2) Normalize
#     cube_norm = minmax_normalize(cube)

#     # 3) IBRA+GSS band selection
#     selected_bands = select_bands_ibra_gss(
#         cube=cube_norm,
#         nbands=num_bands,
#         vif_threshold=vif_threshold,
#         max_distance=max_distance,
#         verbose=True,
#     )
#     print(f"  [IBRA+GSS] Selected bands: {selected_bands}")

#     cube_sel = cube_norm[:, :, selected_bands]

#     # 4) Extract patches
#     X_full, y_full = extract_patches(cube_sel, gt, patch_size)
#     print(f"  Patches shape: {X_full.shape}")

#     # 5) SSL split once (full unlabeled pool)
#     X_l, y_l, X_val, y_val, X_u_full, y_u_masked_full, X_test, y_test = split_ssl_full(
#         X_full,
#         y_full,
#         test_frac=test_frac,
#         labeled_frac=labeled_frac,
#         val_frac=val_frac,
#         random_state=0,
#     )

#     print(f"  Base split → Labeled: {X_l.shape[0]}, Val: {X_val.shape[0]}, Unlabeled pool: {X_u_full.shape[0]}, Test: {X_test.shape[0]}")

#     results_by_frac = {}

#     for frac in unlabeled_fracs:
#         k = int(frac * X_u_full.shape[0])
#         if k <= 0:
#             continue

#         print(f"\n--- Unlabeled fraction {frac:.2f} → {k} samples ---")
#         X_u = X_u_full[:k]
#         y_u_masked = y_u_masked_full[:k]

#         res = run_pseudo_label_ssl(
#             build_model_fn=build_hyper3dnet_lite,
#             X_labeled=X_l,
#             y_labeled=y_l,
#             X_val=X_val,
#             y_val=y_val,
#             X_unlabeled=X_u,
#             y_unlabeled=y_u_masked,
#             X_test=X_test,
#             y_test=y_test,
#             epochs=epochs,
#             batch_size=batch_size,
#             confidence_thresh=confidence_thresh,
#             ssl_iters=ssl_iters,
#         )

#         OA = res["OA"]
#         print(f"  [frac={frac:.2f}] Test OA: {OA:.4f}")
#         results_by_frac[str(frac)] = {
#             "unlabeled_size": int(k),
#             "OA": float(OA),
#             "history": res["history"],
#         }

#     # 6) Save aggregated results
#     os.makedirs(save_dir, exist_ok=True)
#     out_path = os.path.join(
#         save_dir,
#         f"{dataset_name}_ssl_progressive_ibra_gss_{num_bands}bands.json",
#     )

#     payload = {
#         "dataset": dataset_name,
#         "model": "hyper3dnet_lite",
#         "band_selection": "ibra_gss",
#         "selected_bands": list(map(int, selected_bands)),
#         "patch_size": patch_size,
#         "test_frac": test_frac,
#         "labeled_frac": labeled_frac,
#         "val_frac": val_frac,
#         "unlabeled_fracs": unlabeled_fracs,
#         "epochs": epochs,
#         "batch_size": batch_size,
#         "ssl_iters": ssl_iters,
#         "confidence_thresh": confidence_thresh,
#         "results_by_frac": results_by_frac,
#     }

#     with open(out_path, "w") as f:
#         json.dump(payload, f, indent=4)

#     print(f"\nSaved progressive SSL results to: {out_path}")
#     return payload


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--patch", type=int, default=25)
#     parser.add_argument("--num_bands", type=int, default=20)
#     parser.add_argument("--test_frac", type=float, default=0.20)
#     parser.add_argument("--labeled_frac", type=float, default=0.05)
#     parser.add_argument("--val_frac", type=float, default=0.10)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--ssl_iters", type=int, default=5)
#     parser.add_argument("--confidence", type=float, default=0.9)
#     parser.add_argument(
#         "--unlabeled_fracs",
#         type=str,
#         default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85",
#         help="Comma-separated unlabeled fractions",
#     )
#     parser.add_argument("--vif_threshold", type=float, default=10.0)
#     parser.add_argument("--max_distance", type=float, default=5.0)

#     args = parser.parse_args()

#     unlabeled_fracs = tuple(float(x) for x in args.unlabeled_fracs.split(","))

#     run_ssl_progressive_unlabeled(
#         dataset_name=args.dataset,
#         patch_size=args.patch,
#         num_bands=args.num_bands,
#         test_frac=args.test_frac,
#         labeled_frac=args.labeled_frac,
#         val_frac=args.val_frac,
#         unlabeled_fracs=unlabeled_fracs,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         ssl_iters=args.ssl_iters,
#         confidence_thresh=args.confidence,
#         vif_threshold=args.vif_threshold,
#         max_distance=args.max_distance,
#     )


##################### might need to uncomment ....... adding after cross validation 

# experiments/ssl/run_ssl_progressive_unlabeled.py

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
    progressive_unlabeled_subsets,
)
from hyperspectral_insight.ssl.pseudo_label import run_pseudo_label_ssl_fold


def run_ssl_progressive_unlabeled(
    dataset_name: str,
    num_bands: int = 20,
    patch_size: int = 25,
    unlabeled_fracs=(0.10, 0.20, 0.40, 0.60, 0.80),
    ssl_iters: int = 5,
    confidence_thresh: float = 0.9,
    epochs: int = 20,
    batch_size: int = 16,
    save_dir: str = "results/ssl_progressive/",
):
    """
    SSL PROGRESSIVE UNLABELED EXPERIMENT
        - Use fixed 12% labeled pool
        - Use subsets of unlabeled pool with fractions: 10%, 20%, ... 80%
        - Compute TEST accuracy for each fraction

    Output:
        JSON with:
            unlabeled_fraction -> TEST OA
            SSL history per fraction
    """

    print(f"\n=== SSL Progressive Unlabeled Experiment: {dataset_name} ===")

    # Load + normalize
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # Band selection
    selected_bands = select_bands_ibra_gss(
        cube=cube_norm,
        nbands=num_bands,
        vif_threshold=10.0,
        max_distance=5.0,
    )
    print("Selected bands:", selected_bands)

    cube_sel = cube_norm[:, :, selected_bands]

    # Patch extraction
    X_all, y_all = extract_patches(cube_sel, gt, patch_size)

    # SSL main split
    split_dict = ssl_train_val_test_split(
        X_all, y_all,
        test_frac=0.20,
        labeled_frac_within_train=0.12,
        random_state=0,
    )

    X_test = split_dict["X_test"]
    y_test = split_dict["y_test"]

    X_labeled_pool = split_dict["X_labeled_pool"]
    y_labeled_pool = split_dict["y_labeled_pool"]

    X_unlabeled_pool = split_dict["X_unlabeled_pool"]
    y_unlabeled_pool = split_dict["y_unlabeled_pool"]

    # Generate progressive unlabeled subsets
    frac_dict = progressive_unlabeled_subsets(
        X_unlabeled_pool,
        y_unlabeled_pool,
        unlabeled_fracs=unlabeled_fracs,
    )

    results = {}

    for frac in unlabeled_fracs:
        if frac not in frac_dict:
            continue

        print(f"\n--- Running SSL with {int(frac*100)}% of unlabeled ---")
        X_u_frac = frac_dict[frac]["X_u"]

        # SINGLE-FOLD SSL for progressive
        res = run_pseudo_label_ssl_fold(
            build_model_fn=build_hyper3dnet_lite,
            X_labeled_fold=X_labeled_pool,
            y_labeled_fold=y_labeled_pool,
            X_unlabeled_pool=X_u_frac,
            X_test=X_test,
            y_test=y_test,
            epochs_per_iter=epochs,
            batch_size=batch_size,
            confidence_thresh=confidence_thresh,
            ssl_iters=ssl_iters,
        )

        print(f"  Unlabeled {frac} → Test OA = {res['OA_test']:.4f}")
        results[frac] = res

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_ssl_progressive.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "selected_bands": selected_bands,
                "results": results,
            },
            f,
            indent=4,
        )

    print(f"\nSaved progressive SSL results → {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--ssl_iters", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    run_ssl_progressive_unlabeled(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        ssl_iters=args.ssl_iters,
        confidence_thresh=args.conf,
        epochs=args.epochs,
        batch_size=args.batch,
    )
