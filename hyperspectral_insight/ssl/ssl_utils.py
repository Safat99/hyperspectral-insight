import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

def merge_labeled_and_pseudo(
    X_labeled: np.ndarray,
    y_labeled_oh: np.ndarray,
    X_pseudo: np.ndarray,
    y_pseudo: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge newly pseudo-labeled samples into the labeled pool.

    Args:
        X_labeled: (N_l, ...) current labeled data
        y_labeled_oh: (N_l, C) current one-hot labels
        X_pseudo: (N_p, ...) newly pseudo-labeled samples
        y_pseudo: (N_p,) integer pseudo-labels
        n_classes: number of classes

    Returns:
        X_new: concatenated labeled + pseudo-labeled
        y_new_oh: concatenated one-hot labels
    """
    if X_pseudo.shape[0] == 0:
        return X_labeled, y_labeled_oh

    y_pseudo_oh = tf.keras.utils.to_categorical(y_pseudo, num_classes=n_classes)

    X_new = np.concatenate([X_labeled, X_pseudo], axis=0)
    y_new_oh = np.concatenate([y_labeled_oh, y_pseudo_oh], axis=0)

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