import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_subset(X, y, fraction, seed=0):
    """
    Select a stratified subset of (X, y) given a fraction.

    Args:
        X: (N, ...) array of samples
        y: (N,) integer labels
        fraction: float in (0,1), fraction of samples to keep
        seed: random seed

    Returns:
        X_sub, y_sub, X_rest, y_rest
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=fraction,
        random_state=seed,
    )
    idx_sub, idx_rest = next(splitter.split(X, y))

    return X[idx_sub], y[idx_sub], X[idx_rest], y[idx_rest]


def make_ssl_split(X, y, labeled_frac=0.05, seed=0):
    """
    Create a single SSL split:
        - stratified labeled subset
        - remaining unlabeled subset (labels masked as -1)

    Args:
        X: (N, ...) samples
        y: (N,) integer labels
        labeled_frac: e.g. 0.05 for 5% labeled
        seed: RNG seed

    Returns:
        X_l, y_l, X_u, y_u_masked
    """
    X_l, y_l, X_rest, y_rest = stratified_subset(X, y, labeled_frac, seed)
    y_u_masked = np.full_like(y_rest, fill_value=-1)

    return X_l, y_l, X_rest, y_u_masked

def progressive_unlabeled_splits(
    X,
    y,
    labeled_frac=0.05,
    unlabeled_fracs=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85),
    seed=0,
):
    """
    Create multiple SSL splits with a fixed labeled set and
    progressively larger unlabeled subsets.

    This is for performance-vs-unlabeled curves.

    Args:
        X, y: full dataset
        labeled_frac: fraction used as labeled (stratified)
        unlabeled_fracs: sequence of floats in (0,1). Each is a fraction
                         of the UNLABELED POOL (not whole dataset).
        seed: RNG seed

    Returns:
        dict mapping frac -> (X_l, y_l, X_u_frac, y_u_masked_frac)
    """
    X_l, y_l, X_rest, y_rest = stratified_subset(X, y, labeled_frac, seed)

    results = {}
    N_u = len(X_rest)

    for frac in unlabeled_fracs:
        k = int(frac * N_u)
        if k <= 0:
            continue

        X_u_frac = X_rest[:k]
        y_u_masked_frac = np.full(k, -1, dtype=y_rest.dtype)

        results[frac] = (X_l, y_l, X_u_frac, y_u_masked_frac)

    return results

def supervised_full_split(X, y):
    """
    Supervised baseline: use all samples as labeled.

    Returns:
        X_full, y_full
    """
    return X, y

