# hyperspectral_insight/ssl/split.py

import numpy as np
from typing import Dict, Tuple, Iterable, List
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def ssl_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.20,
    labeled_frac_within_train: float = 0.12,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Single global split for SSL + CV.

    1) Hold out a TEST set (test_frac of full data) – never used in training.
    2) From remaining (1 - test_frac), take a LABELED POOL (labeled_frac_within_train)
       and treat the rest as UNLABELED POOL.
    3) K-fold CV will be done ONLY on the LABELED POOL.
       The UNLABELED POOL is shared across all folds.

    Args:
        X: (N, ...) patch data.
        y: (N,) integer labels.
        test_frac: fraction of full data kept as test.
        labeled_frac_within_train: fraction of the *80%* training portion
                                   that is labeled. Example:
                                   - test_frac=0.20
                                   - labeled_frac_within_train=0.12
                                   => 0.8 * 0.12 = 0.096 ≈ 9.6% of full data labeled.
        random_state: RNG seed.

    Returns:
        dict with:
            - X_test, y_test
            - X_labeled_pool, y_labeled_pool    (≈ 12% of 80% = 9.6% of full)
            - X_unlabeled_pool, y_unlabeled_pool (≈ 68% of full)
    """

    X = np.asarray(X)
    y = np.asarray(y)

    # --------------------------------------
    # 1) TEST split: 20% held-out
    # --------------------------------------
    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=random_state,
    )
    idx_train, idx_test = next(sss_test.split(X, y))

    X_train_full = X[idx_train]
    y_train_full = y[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    # --------------------------------------
    # 2) Labeled vs unlabeled within training portion
    # --------------------------------------
    sss_lab = StratifiedShuffleSplit(
        n_splits=1,
        train_size=labeled_frac_within_train,
        random_state=random_state,
    )
    idx_labeled, idx_unlabeled = next(sss_lab.split(X_train_full, y_train_full))

    X_labeled_pool = X_train_full[idx_labeled]
    y_labeled_pool = y_train_full[idx_labeled]

    X_unlabeled_pool = X_train_full[idx_unlabeled]
    y_unlabeled_pool = y_train_full[idx_unlabeled]  # kept for analysis only; not used in training

    print(
        "[SSL SPLIT] Sizes:\n"
        f"  Total:          {len(X)}\n"
        f"  Test:           {len(X_test)} ({len(X_test)/len(X):.3f})\n"
        f"  Labeled pool:   {len(X_labeled_pool)} ({len(X_labeled_pool)/len(X):.3f})\n"
        f"  Unlabeled pool: {len(X_unlabeled_pool)} ({len(X_unlabeled_pool)/len(X):.3f})"
    )

    return {
        "X_test": X_test,
        "y_test": y_test,
        "X_labeled_pool": X_labeled_pool,
        "y_labeled_pool": y_labeled_pool,
        "X_unlabeled_pool": X_unlabeled_pool,
        "y_unlabeled_pool": y_unlabeled_pool,
    }


def ssl_kfold_indices_for_labeled(
    y_labeled_pool: np.ndarray,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-fold ONLY on the labeled pool (12% portion).

    Args:
        y_labeled_pool: (N_labeled,)
        n_splits: K in K-fold.
        shuffle: whether to shuffle.
        random_state: RNG seed.

    Yields:
        (train_idx, val_idx) for each fold, indices relative to the labeled pool.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for tr_idx, val_idx in skf.split(np.zeros_like(y_labeled_pool), y_labeled_pool):
        yield tr_idx, val_idx


def progressive_unlabeled_subsets(
    X_unlabeled_pool: np.ndarray,
    y_unlabeled_pool: np.ndarray,
    unlabeled_fracs: List[float],
    random_state: int = 0,
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Create different unlabeled subsets for progressive SSL experiments.

    All subsets are drawn from a fixed unlabeled pool (after test split).
    IMPORTANT: fractions are taken w.r.t. the unlabeled pool size, not full dataset.

    Args:
        X_unlabeled_pool: (N_u, ...)
        y_unlabeled_pool: (N_u,)
        unlabeled_fracs: e.g. [0.1, 0.2, 0.4, 0.6, 0.8]
        random_state: RNG seed.

    Returns:
        dict[frac] = {
            "X_u": X_subset,
            "y_u": y_subset
        }
    """

    rng = np.random.RandomState(random_state)
    N_u = len(X_unlabeled_pool)
    indices = np.arange(N_u)

    frac_dict = {}

    for frac in unlabeled_fracs:
        k = int(frac * N_u)
        if k <= 0:
            continue
        # random subset of unlabeled pool
        idx_frac = rng.choice(indices, size=k, replace=False)
        frac_dict[frac] = {
            "X_u": X_unlabeled_pool[idx_frac],
            "y_u": y_unlabeled_pool[idx_frac],
        }

    return frac_dict


def supervised_full_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supervised baseline: use all samples as labeled.
    Kept for compatibility with non-SSL baselines.
    """
    return X, y
