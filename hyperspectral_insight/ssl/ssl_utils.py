import numpy as np
from typing import Dict, Tuple

def merge_labeled_and_pseudo(
    X_l: np.ndarray,
    y_l_oh: np.ndarray,
    X_p: np.ndarray,
    y_p: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge labeled data (already one-hot) with new pseudo-labeled samples.

    Args:
        X_l: (N_l, ...) current training data
        y_l_oh: (N_l, C) one-hot labels
        X_p: (N_p, ...) pseudo-labeled samples
        y_p: (N_p,) integer pseudo-labels
        n_classes: number of classes

    Returns:
        X_new, y_new_oh
    """
    if X_p.shape[0] == 0:
        return X_l, y_l_oh

    y_p_oh = np.eye(n_classes, dtype=np.float32)[y_p]

    X_new = np.concatenate([X_l, X_p], axis=0)
    y_new_oh = np.concatenate([y_l_oh, y_p_oh], axis=0)

    return X_new, y_new_oh


def compute_class_distribution(y: np.ndarray) -> Dict[int, int]:
    """
    Compute counts per class from integer label vector.

    Args:
        y: (N,) integer labels

    Returns:
        dict {class_id: count}
    """
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def select_balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select at most max_per_class samples per class from (X, y).

    Useful for:
        - balancing pseudo-labeled sets
        - avoiding dominance of majority classes

    Args:
        X: (N, ...) samples
        y: (N,) integer labels
        max_per_class: upper cap per class
        seed: RNG seed

    Returns:
        X_bal, y_bal
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) > max_per_class:
            idx_c = rng.choice(idx_c, size=max_per_class, replace=False)
        indices.append(idx_c)

    indices = np.concatenate(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]