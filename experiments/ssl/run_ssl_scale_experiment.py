import os
import json
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.band_selection.ibra_gss import select_bands_ibra_gss
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite

from hyperspectral_insight.ssl.pseudo_label import run_pseudo_label_ssl_fold
from hyperspectral_insight.evaluation.cross_validation import cap_samples_per_class


# ============================================================
# Stratified subset sampling
# ============================================================
def stratified_sample(X, y, frac, random_state=0):
    if frac >= 1.0:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac, random_state=random_state)
    for idx, _ in sss.split(X, y):
        return X[idx], y[idx]


# ============================================================
# Split subset into labeled + unlabeled
# ============================================================
def make_labeled_unlabeled_split(X, y, labeled_ratio=0.2, random_state=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - labeled_ratio, random_state=random_state)
    for lab_idx, unlab_idx in sss.split(X, y):
        return X[lab_idx], y[lab_idx], X[unlab_idx]


# ============================================================
# MAIN SSL SCALING EXPERIMENT
# ============================================================
def run_ssl_scaling_experiment(
    dataset_name,
    subset_fracs=[0.10, 0.20, 0.40, 0.60, 0.80],
    labeled_ratio=0.20,
    num_bands=5,
    n_splits=10,
    ssl_iters=3,
    confidence_thresh=0.9,
    epochs_baseline=30,
    optimizer="adam",
    patch_size=17,
    batch_size=32,
    lr=1e-4,
    rho=0.95,
    epsilon=1e-7,
    max_samples_per_class=None,
    random_state=0,
    save_dir="results/scaling_ssl/final/",
):

    print("\n========== SSL SCALING EXPERIMENT ==========")
    print(f"Dataset: {dataset_name}")
    print(f"Baseline epoch budget: {epochs_baseline}")
    print(f"SSL iterations: {ssl_iters}")

    # ----------------------------------------------------------
    # Load + Normalize
    # ----------------------------------------------------------
    cube, gt = load_dataset(dataset_name)
    cube = minmax_normalize(cube)

    # ----------------------------------------------------------
    # IBRA+GSS Band Selection
    # ----------------------------------------------------------
    selected_bands = select_bands_ibra_gss(
        cube=cube,
        nbands=num_bands,
        vif_threshold=10.0,
        max_distance=5.0,
        verbose=False,
    )
    cube = cube[:, :, selected_bands]

    # ----------------------------------------------------------
    # Patch Extraction
    # ----------------------------------------------------------
    X_all, y_all = extract_patches(
        cube, gt,
        win=patch_size,
        drop_label0=True,
        max_samples_per_class=None
    )

    results_all = {}

    # ----------------------------------------------------------
    # Model Builder
    # ----------------------------------------------------------
    def model_fn(input_shape, n_classes):
        if optimizer.lower() == "adam":
            return build_hyper3dnet_lite(input_shape, n_classes, optimizer_name="adam", lr=lr)
        else:
            return build_hyper3dnet_lite(input_shape, n_classes, optimizer_name="adadelta", rho=rho, epsilon=epsilon)

    # ----------------------------------------------------------
    # Loop over labeled data fractions
    # ----------------------------------------------------------
    for frac in subset_fracs:
        print(f"\n===== Subset {int(frac*100)}% =====")

        X_sub, y_sub = stratified_sample(X_all, y_all, frac, random_state)
        X_lab, y_lab, X_unlab = make_labeled_unlabeled_split(
            X_sub, y_sub, labeled_ratio=labeled_ratio, random_state=random_state
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        baseline_oa, baseline_f1 = [], []
        ssl_oa, ssl_f1 = [], []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_lab, y_lab), 1):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")

            X_train = X_lab[tr_idx]
            y_train = y_lab[tr_idx]
            X_val = X_lab[val_idx]
            y_val = y_lab[val_idx]

            # Apply SAME class cap as supervised experiments
            X_train, y_train = cap_samples_per_class(
                X_train, y_train, max_samples_per_class, seed=fold_idx
            )

            # ---------------- Baseline ----------------
            baseline_res = run_pseudo_label_ssl_fold(
                model_fn,
                X_train,
                y_train,
                None,
                X_val,
                y_val,
                epochs_per_iter=epochs_baseline,
                batch_size=batch_size,
                ssl_iters=0,
                confidence_thresh=None,
                val_frac=0.0,
            )
            baseline_oa.append(baseline_res["OA_test"])
            baseline_f1.append(baseline_res["F1_test"])

            # ---------------- SSL ----------------
            ssl_res = run_pseudo_label_ssl_fold(
                model_fn,
                X_train,
                y_train,
                X_unlab,
                X_val,
                y_val,
                epochs_per_iter=epochs_baseline,  # budget handled internally now
                batch_size=batch_size,
                ssl_iters=ssl_iters,
                confidence_thresh=confidence_thresh,
                val_frac=0.0,
            )
            ssl_oa.append(ssl_res["OA_test"])
            ssl_f1.append(ssl_res["F1_test"])

        results_all[frac] = {
            "baseline_oa": baseline_oa,
            "ssl_oa": ssl_oa,
            "baseline_f1": baseline_f1,
            "ssl_f1": ssl_f1,

            "mean_baseline_oa": float(np.mean(baseline_oa)),
            "std_baseline_oa": float(np.std(baseline_oa, ddof=1)),
            "mean_ssl_oa": float(np.mean(ssl_oa)),
            "std_ssl_oa": float(np.std(ssl_oa, ddof=1)),

            "mean_baseline_f1": float(np.mean(baseline_f1)),
            "std_baseline_f1": float(np.std(baseline_f1, ddof=1)),
            "mean_ssl_f1": float(np.mean(ssl_f1)),
            "std_ssl_f1": float(np.std(ssl_f1, ddof=1)),

            "selected_bands": list(map(int, selected_bands)),
        }

        print(f"Baseline F1 mean: {np.mean(baseline_f1):.4f}")
        print(f"SSL F1 mean     : {np.mean(ssl_f1):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_ssl_scaling_results.json")

    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\nSaved results to {out_path}")
    return results_all


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--ssl_iters", type=int, default=3)
    parser.add_argument("--epochs_baseline", type=int, default=30)
    parser.add_argument("--confidence_thresh", type=float, default=0.9)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    run_ssl_scaling_experiment(
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        optimizer=args.optimizer,
        rho=args.rho,
        epsilon=args.epsilon,
        n_splits=args.n_splits,
        ssl_iters=args.ssl_iters,
        epochs_baseline=args.epochs_baseline,
        confidence_thresh=args.confidence_thresh,
        max_samples_per_class=args.max_samples,
    )
