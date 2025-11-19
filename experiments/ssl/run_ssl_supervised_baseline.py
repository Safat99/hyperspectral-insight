# # experiments/ssl/run_ssl_supervised_baseline.py

# import os
# import json
# import numpy as np

# from sklearn.model_selection import train_test_split

# from hyperspectral_insight.data.loaders import load_dataset
# from hyperspectral_insight.data.patches import extract_patches
# from hyperspectral_insight.data.normalization import minmax_normalize

# from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
# from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
# from hyperspectral_insight.evaluation.metrics import compute_metrics


# def supervised_split(
#     X,
#     y,
#     test_frac=0.20,
#     val_frac=0.10,
#     random_state=0,
# ):
#     """
#     Supervised train/val/test split (all labeled).

#     Returns:
#         X_train, y_train,
#         X_val, y_val,
#         X_test, y_test
#     """
#     # 1) Test split
#     X_trainval, X_test, y_trainval, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_frac,
#         stratify=y,
#         random_state=random_state,
#     )

#     # 2) Validation split from trainval
#     val_frac_adj = val_frac / (1.0 - test_frac)  # relative fraction
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_trainval,
#         y_trainval,
#         test_size=val_frac_adj,
#         stratify=y_trainval,
#         random_state=random_state,
#     )

#     return X_train, y_train, X_val, y_val, X_test, y_test


# def run_supervised_baseline(
#     dataset_name: str,
#     patch_size: int = 25,
#     epochs: int = 50,
#     batch_size: int = 32,
#     test_frac: float = 0.20,
#     val_frac: float = 0.10,
#     use_ibra_gss: bool = False,
#     num_bands: int = 20,
#     vif_threshold: float = 10.0,
#     max_distance: float = 5.0,
#     save_dir: str = "results/ssl/",
# ):
#     """
#     Supervised upper-bound baseline using Hyper3DNet-Lite.

#     Cases:
#       - use_ibra_gss=False → full-band Hyper3DNet-Lite
#       - use_ibra_gss=True  → IBRA+GSS band selection, then Hyper3DNet-Lite

#     Saves:
#       JSON with metrics on test set.
#     """

#     print(f"\n=== Supervised Baseline ({dataset_name}) ===")
#     print(f"  Model: Hyper3DNet-Lite")
#     print(f"  Band selection: {'IBRA+GSS' if use_ibra_gss else 'FULL-BANDS'}")

#     # 1) Load dataset
#     cube, gt = load_dataset(dataset_name)

#     # 2) Normalize per band
#     cube_norm = minmax_normalize(cube)

#     # 3) Optional IBRA+GSS band-selection
#     selected_bands = None
#     if use_ibra_gss:
#         print(f"Running IBRA+GSS band selection for {num_bands} bands...")
#         selected_bands = select_bands_ibra_gss(
#             cube=cube_norm,
#             nbands=num_bands,
#             vif_threshold=vif_threshold,
#             max_distance=max_distance,
#             verbose=True,
#         )
#         print(f"  Selected bands: {selected_bands}")
#         cube_norm = cube_norm[:, :, selected_bands]

#     # 4) Extract patches (drop label 0 inside extract_patches implementation)
#     X, y = extract_patches(cube_norm, gt, patch_size)

#     print(f"  Patches shape: {X.shape}, labels shape: {y.shape}")
#     print(f"  #classes (excluding 0): {int(y.max()) + 1}")

#     # 5) Supervised split
#     X_train, y_train, X_val, y_val, X_test, y_test = supervised_split(
#         X,
#         y,
#         test_frac=test_frac,
#         val_frac=val_frac,
#         random_state=0,
#     )

#     n_classes = int(y_train.max() + 1)
#     input_shape = X_train.shape[1:]

#     # 6) Build model
#     model = build_hyper3dnet_lite(input_shape=input_shape, n_classes=n_classes)
#     y_train_oh = np.eye(n_classes)[y_train]
#     y_val_oh = np.eye(n_classes)[y_val]

#     # 7) Train
#     history = model.fit(
#         X_train,
#         y_train_oh,
#         validation_data=(X_val, y_val_oh),
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=1,
#     )

#     # 8) Evaluate on test
#     y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
#     metrics = compute_metrics(y_test, y_test_pred)

#     print("\n[Supervised Baseline] Test metrics:")
#     for k, v in metrics.items():
#         print(f"  {k}: {v:.4f}")

#     # 9) Save JSON
#     os.makedirs(save_dir, exist_ok=True)

#     tag = "supervised_lite_fullbands"
#     if use_ibra_gss:
#         tag = f"supervised_lite_ibra_gss_{num_bands}bands"

#     out_path = os.path.join(save_dir, f"{dataset_name}_{tag}.json")

#     payload = {
#         "dataset": dataset_name,
#         "model": "hyper3dnet_lite",
#         "band_selection": "ibra_gss" if use_ibra_gss else "full_bands",
#         "selected_bands": selected_bands,
#         "patch_size": patch_size,
#         "epochs": epochs,
#         "batch_size": batch_size,
#         "test_frac": test_frac,
#         "val_frac": val_frac,
#         "metrics": metrics,
#     }

#     with open(out_path, "w") as f:
#         json.dump(payload, f, indent=4)

#     print(f"Saved supervised baseline results to: {out_path}")
#     return payload


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--patch", type=int, default=25)
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--test_frac", type=float, default=0.20)
#     parser.add_argument("--val_frac", type=float, default=0.10)
#     parser.add_argument("--use_ibra_gss", action="store_true")
#     parser.add_argument("--num_bands", type=int, default=20)
#     parser.add_argument("--vif_threshold", type=float, default=10.0)
#     parser.add_argument("--max_distance", type=float, default=5.0)

#     args = parser.parse_args()

#     run_supervised_baseline(
#         dataset_name=args.dataset,
#         patch_size=args.patch,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         test_frac=args.test_frac,
#         val_frac=args.val_frac,
#         use_ibra_gss=args.use_ibra_gss,
#         num_bands=args.num_bands,
#         vif_threshold=args.vif_threshold,
#         max_distance=args.max_distance,
#     )


############# updated after adding cross validation .. might need to edit later 

# experiments/ssl/run_supervised_full_labeled_baseline.py

import os
import json
import numpy as np

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches

from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation


def run_supervised_full_labeled_baseline(
    dataset_name: str,
    num_bands: int = 20,
    patch_size: int = 25,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    save_dir: str = "results/ssl/",
):
    """
    SUPERVISED BASELINE:
        - Use ALL samples as fully labeled
        - IBRA-GSS band selection
        - Hyper3DNet-Lite supervised learning
        - Normal stratified K-fold CV

    This forms the "upper bound" that SSL attempts to approximate.
    """

    print(f"\n=== Supervised FULL-LABELED Baseline on {dataset_name} ===")

    # Load + normalize
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # Band selection
    selected_bands = select_bands_ibra_gss(
        cube=cube_norm,
        nbands=num_bands,
        vif_threshold=10.0,
        max_distance=5.0,
        verbose=False,
    )
    print(f"Selected bands: {selected_bands}")

    cube_sel = cube_norm[:, :, selected_bands]

    # Patch extraction
    X_all, y_all = extract_patches(cube_sel, gt, patch_size)
    print(f"  Patches: {X_all.shape}, Labels: {y_all.shape}")
    print(f"  Classes: {y_all.max()+1}")

    # Supervised K-fold CV
    results = kfold_cross_validation(
        model_fn=build_hyper3dnet_lite,
        X=X_all,
        y=y_all,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        random_state=0,
        verbose=0,
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_supervised_full_baseline.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "selected_bands": selected_bands,
                "cv_results": results,
            },
            f,
            indent=4,
        )

    print(f"Saved supervised baseline → {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_supervised_full_labeled_baseline(
        dataset_name=args.dataset,
        num_bands=args.num_bands,
        patch_size=args.patch,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
