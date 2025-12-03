import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support
)
from tensorflow.keras.utils import to_categorical

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.models.hyper3dnet import build_hyper3dnet


def run_h3dnet_with_pca_cv(
    dataset_name: str,
    patch_size: int = 25,
    var_thresh: float = 0.99,
    max_components: int = 30,
    n_splits: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = "results/hyper3dnet/",
):
    """
    Hyper3DNet + PCA applied *inside* each fold.
    PCA is fit ONLY on training-set patches (per fold).
    PCA components capped at `max_components`.

    Saves:
        - JSON metrics
        - CSV per-fold PCA component counts
        - NPY fold histories
    """

    print(f"\n=== Hyper3DNet + PCA (top ≤ {max_components}) on {dataset_name} ===")

    # 1. Load dataset
    cube, gt = load_dataset(dataset_name)

    # 2. Normalize
    cube_norm = minmax_normalize(cube)

    # 3. Extract labeled patches before PCA
    X_full, y_full = extract_patches(cube_norm, gt, patch_size)
    n_classes = int(y_full.max() + 1)

    print(f"Total patches: {len(X_full)}  |  Classes: {n_classes}")

    # one-hot labels
    y_onehot = to_categorical(y_full, num_classes=n_classes)

    # prepare CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    fold_metrics = []
    pca_components_list = []
    fold_histories = []
    fold_pca_info = []


    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):

        print(f"\n===== Fold {fold}/{n_splits} =====")

        # ---- split ----
        Xtr_raw, Xval_raw = X_full[tr_idx], X_full[val_idx]
        ytr, yval = y_onehot[tr_idx], y_onehot[val_idx]

        Ntr = Xtr_raw.shape[0]
        H, W, B, _ = Xtr_raw.shape[1:]

        # ---- fit PCA on training spectra only ----
        Xtrain_flat = Xtr_raw.reshape(-1, B)  # (N * H * W, B)

        pca = PCA(n_components=min(max_components, B))
        Xtrain_pca = pca.fit_transform(Xtrain_flat)

        # choose components based on variance threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        D = np.argmax(cumvar >= var_thresh) + 1
        D = min(D, max_components)
        
        # ---- Save full PCA info for interpretability ----
        # Components: (D, B)
        # Explained variance: (D,)
        # Mean: (B,)
        components = pca.components_[:D]
        explained = pca.explained_variance_ratio_[:D]
        mean_vec = pca.mean_
        band_importance = np.sum(np.abs(components), axis=0)
        fold_pca_info.append({
            "fold": fold,
            "D": int(D),
            "components": components.tolist(),
            "explained_variance_ratio": explained.tolist(),
            "mean": mean_vec.tolist(),
            "band_importance": band_importance.tolist(),
        })

        print(f"[Fold {fold}] Using {D} PCA components "
              f"({cumvar[D-1]*100:.2f}% variance)")
        pca_components_list.append(D)
        
        

        # ---- Project both train & val ----
        Xtrain_pca = Xtrain_pca[:, :D]
        Xval_flat = Xval_raw.reshape(-1, B)
        Xval_pca = pca.transform(Xval_flat)[:, :D]

        # ---- Reshape back to patches ----
        Xtr = Xtrain_pca.reshape(Ntr, H, W, D, 1)
        Xval = Xval_pca.reshape(Xval_raw.shape[0], H, W, D, 1)

        # ---- Build Hyper3DNet ----
        model = build_hyper3dnet(input_shape=(H, W, D, 1), n_classes=n_classes, lr=lr)
        

        # ---- Train ----
        history = model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save history dict for this fold
        history_dict = {k: list(v) for k, v in history.history.items()}
        fold_histories.append(history_dict)

        # ---- Evaluate ----
        ypred = model.predict(Xval)
        ypred_cls = np.argmax(ypred, axis=1)
        ytrue_cls = np.argmax(yval, axis=1)

        oa = accuracy_score(ytrue_cls, ypred_cls)
        kappa = cohen_kappa_score(ytrue_cls, ypred_cls)
        prec, rec, f1, _ = precision_recall_fscore_support(
            ytrue_cls, ypred_cls, average="macro"
        )

        print(f"[Fold {fold}] OA={oa:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}")

        fold_metrics.append({
            "OA": float(oa),
            "Kappa": float(kappa),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "PCA_components": int(D),
        })

    # ----------------------------------------------------------
    # aggregate
    keys = ["OA", "Kappa", "Precision", "Recall", "F1"]
    mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    std_metrics = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}

    # ----------------------------------------------------------
    # save
    os.makedirs(save_dir, exist_ok=True)

    out_json = os.path.join(save_dir, f"{dataset_name}_h3dnet_pca_cv.json")
    with open(out_json, "w") as f:
        json.dump({
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "pca_components": pca_components_list,
            "pca_details": fold_pca_info,
        }, f, indent=4)

    # also CSV for PCA component counts
    pd.DataFrame({
        "fold": np.arange(1, n_splits+1),
        "components": pca_components_list
    }).to_csv(
        os.path.join(save_dir, f"{dataset_name}_h3dnet_pca_components.csv"),
        index=False
    )
    
    # save histories as .npy
    out_hist = os.path.join(save_dir, f"{dataset_name}_h3dnet_pca_histories.npy")
    np.save(out_hist, fold_histories, allow_pickle=True)
    
    
    pca_components_out = os.path.join(save_dir, f"{dataset_name}_pca_loadings.npy")
    np.save(pca_components_out,
            np.array([f["components"] for f in fold_pca_info], dtype=object),
            allow_pickle=True)

    # Save explained variance ratios
    pca_var_out = os.path.join(save_dir, f"{dataset_name}_pca_variance.npy")
    np.save(pca_var_out,
            np.array([f["explained_variance_ratio"] for f in fold_pca_info], dtype=object),
            allow_pickle=True)
    

    print(f"PCA loadings saved to: {pca_components_out}")
    print(f"PCA variance saved to: {pca_var_out}")

    print("\n=== PCA-based Hyper3DNet DONE ===")
    print(f"Results saved to: {out_json}")
    print(f"Histories saved to: {out_hist}")
    print(f"Mean metrics: {mean_metrics}")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "pca_components": pca_components_list,
        "histories": fold_histories,
        "pca_details": fold_pca_info,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch", type=int, default=25)
    parser.add_argument("--var_thresh", type=float, default=0.99)
    parser.add_argument("--max_components", type=int, default=30)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    run_h3dnet_with_pca_cv(
        dataset_name=args.dataset,
        patch_size=args.patch,
        var_thresh=args.var_thresh,
        max_components=args.max_components,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )