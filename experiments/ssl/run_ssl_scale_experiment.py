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



# ============================================================
# Utility: stratified sample of dataset
# ============================================================
def stratified_sample(X, y, frac, random_state=0):
    if frac >= 1.0:
        return X, y

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - frac,
        random_state=random_state,
    )
    for idx, _ in sss.split(X, y):
        return X[idx], y[idx]


# ============================================================
# Utility: split subset into labeled + unlabeled
# ============================================================
def make_labeled_unlabeled_split(X, y, labeled_ratio=0.2, random_state=0):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - labeled_ratio,
        random_state=random_state,
    )
    for lab_idx, unlab_idx in sss.split(X, y):
        return X[lab_idx], y[lab_idx], X[unlab_idx]


# ============================================================
# MAIN EXPERIMENT SCRIPT
# ============================================================
def run_ssl_scaling_experiment(
    dataset_name,
    subset_fracs=[0.10, 0.20, 0.40, 0.60, 0.85],
    labeled_ratio=0.20,
    num_bands=5,
    patch_size=25,
    n_splits=5,
    ssl_iters=3,
    confidence_thresh=0.9,
    epochs_baseline=30,
    epochs_ssl_per_iter=15,
    batch_size: int=128,
    save_dir="results/scaling_ssl/updated/",
    random_state=0,
    lr: float = 5e-4,
     max_samples_per_class: int = None,
):

    print("\n==============================================")
    print("         SSL SCALING EXPERIMENT START")
    print("==============================================\n")

    # ----------------------------------------------------------
    # 1. Load + Normalize Dataset
    # ----------------------------------------------------------
    cube, gt = load_dataset(dataset_name)
    cube_norm = minmax_normalize(cube)

    # ----------------------------------------------------------
    # 2. Band Selection (unsupervised)
    # ----------------------------------------------------------
    print("Running IBRA+GSS band selection...")
    selected_bands = select_bands_ibra_gss(
        cube=cube_norm,
        nbands=num_bands,
        vif_threshold=10.0,
        max_distance=5.0,
        verbose=False,
    )
    cube_sel = cube_norm[:, :, selected_bands]

    # ----------------------------------------------------------
    # 3. Patch Extraction
    # ----------------------------------------------------------
    # X_all, y_all = extract_patches(cube_sel, gt, patch_size)
    X_all, y_all = extract_patches(
        cube_sel, gt,
        win=patch_size,
        drop_label0=True,
        max_samples_per_class=max_samples_per_class
    )
    
    print(f"Total patches: {X_all.shape}, labels: {y_all.shape}")

    results_all = {}

    # ----------------------------------------------------------
    # LOOP THROUGH SUBSET FRACTIONS
    # ----------------------------------------------------------
    for frac in subset_fracs:
        print(f"\n\n==============================")
        print(f" SUBSET = {frac*100:.0f}%")
        print("==============================")

        # ----------------------------------------------------------
        # 4. Stratified sample
        # ----------------------------------------------------------
        X_sub, y_sub = stratified_sample(
            X_all, y_all, frac=frac, random_state=random_state
        )
        print(f"Subset size: {len(X_sub)} samples")

        # ----------------------------------------------------------
        # Labeled/Unlabeled split
        # ----------------------------------------------------------
        X_lab, y_lab, X_unlab = make_labeled_unlabeled_split(
            X_sub, y_sub, labeled_ratio=labeled_ratio, random_state=random_state
        )
        print(f"Labeled = {len(X_lab)}, Unlabeled = {len(X_unlab)}")

        # ----------------------------------------------------------
        # Prepare K-fold CV
        # ----------------------------------------------------------
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        baseline_fold_scores = []
        ssl_fold_scores = []

        
        def model_fn(input_shape, n_classes):
            return build_hyper3dnet_lite(input_shape, n_classes, lr=lr)
        
        # ----------------------------------------------------------
        # K-FOLD LOOP
        # ----------------------------------------------------------
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_lab, y_lab), start=1):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")

            X_lab_tr = X_lab[tr_idx]
            y_lab_tr = y_lab[tr_idx]
            X_lab_val = X_lab[val_idx]
            y_lab_val = y_lab[val_idx]

            # ----------------------------------------------------------
            # BASELINE (supervised only)
            # ----------------------------------------------------------
            print("Training BASELINE...")
            baseline_res = run_pseudo_label_ssl_fold(
                build_model_fn=model_fn,
                X_labeled_fold=X_lab_tr,
                y_labeled_fold=y_lab_tr,
                X_unlabeled_pool=None,
                X_test=X_lab_val,   # validation as test for CV
                y_test=y_lab_val,
                epochs_per_iter=epochs_baseline,
                batch_size=batch_size,
                ssl_iters=1,                # no SSL iterations
                confidence_thresh=None,     # not needed
                val_frac=0.0,
            )
            baseline_fold_scores.append(baseline_res["OA_test"])

            # ----------------------------------------------------------
            # SSL MODEL (L + U)
            # ----------------------------------------------------------
            print("Training SSL...")
            ssl_res = run_pseudo_label_ssl_fold(
                build_model_fn=model_fn,
                X_labeled_fold=X_lab_tr,
                y_labeled_fold=y_lab_tr,
                X_unlabeled_pool=X_unlab,
                X_test=X_lab_val,
                y_test=y_lab_val,
                epochs_per_iter=epochs_ssl_per_iter,
                batch_size=batch_size,
                ssl_iters=ssl_iters,
                confidence_thresh=confidence_thresh,
                val_frac=0.0,
            )
            ssl_fold_scores.append(ssl_res["OA_test"])


        # ----------------------------------------------------------
        # Aggregate fold statistics (mean, std, CI)
        # ----------------------------------------------------------
        K = n_splits

        # Baseline stats
        mean_b = float(np.mean(baseline_fold_scores))
        std_b  = float(np.std(baseline_fold_scores, ddof=1))
        sem_b  = std_b / np.sqrt(K)
        ci_b   = (mean_b - 1.96 * sem_b, mean_b + 1.96 * sem_b)

        # SSL stats
        mean_s = float(np.mean(ssl_fold_scores))
        std_s  = float(np.std(ssl_fold_scores, ddof=1))
        sem_s  = std_s / np.sqrt(K)
        ci_s   = (mean_s - 1.96 * sem_s, mean_s + 1.96 * sem_s)

        results_all[frac] = {
            "baseline_scores": baseline_fold_scores,
            "ssl_scores": ssl_fold_scores,

            "mean_baseline": mean_b,
            "mean_ssl": mean_s,

            "std_baseline": std_b,
            "std_ssl": std_s,

            "ci_baseline": ci_b,
            "ci_ssl": ci_s,

            "labeled_ratio": labeled_ratio,
            "selected_bands": list(map(int, selected_bands)),
        }

        print("\nResults for this subset:")
        print("  Baseline mean:", mean_b, "CI:", ci_b)
        print("  SSL mean     :", mean_s, "CI:", ci_s)

    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"{dataset_name}_b{batch_size}_n{num_bands}_ssl_scaling_results.json"
    )

    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\nSaved ALL results to {out_path}\n")
    return results_all




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--n_splits", type=int, default=10)
    
    args = parser.parse_args()

    run_ssl_scaling_experiment(
        dataset_name=args.dataset,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples,
        n_splits=args.n_splits,
    )
