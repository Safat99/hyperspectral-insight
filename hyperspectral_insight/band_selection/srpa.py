import numpy as np
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

def srpa_scores(
    X: np.ndarray,
    y: np.ndarray,
    penalty_lambda: float = 0.3,
) -> np.ndarray:
    """
    Compute SRPA scores for each band.

    Args:
        X: (N, B) spectral samples.
        y: (N,) labels.
        penalty_lambda: redundancy penalty weight.

    Returns:
        srpa_score: (B,) array of scores (higher is better).
    """
    B = X.shape[1]

    # RF importances (attention weights)
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X, y)
    attention = rf.feature_importances_

    # Redundancy via correlation
    corr = np.zeros(B, dtype=np.float32)
    for i in range(B):
        vals = []
        for j in range(B):
            if j == i:
                continue
            r, _ = pearsonr(X[:, i], X[:, j])
            vals.append(abs(r))
        corr[i] = np.mean(vals) if vals else 0.0

    srpa_score = attention - penalty_lambda * corr
    return srpa_score


def select_bands_srpa(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
    penalty_lambda: float = 0.3,
) -> List[int]:
    """
    Select bands via SRPA from an image cube.

    Args:
        cube: (H, W, B)
        gt:   (H, W)
        k:    number of bands
        penalty_lambda: redundancy penalty

    Returns:
        List of selected band indices.
    """
    if cube.ndim == 3:
        H, W, B = cube.shape
        cube_flat = cube.reshape(-1, B)
        mask_flat = gt.flatten()
        valid = mask_flat > 0
        X = cube_flat[valid]
        y = mask_flat[valid]
    else:
        raise ValueError("cube must be 3D (H,W,B) for select_bands_srpa")

    scores = srpa_scores(X, y, penalty_lambda=penalty_lambda)
    selected = np.argsort(scores)[::-1][:k]
    return [int(b) for b in selected]


def run_srpa_pipeline(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
    penalty_lambda: float = 0.3,
    verbose: bool = True,
) -> Tuple[List[int], float, float]:
    """
    Run SRPA + quick RF evaluation (on selected bands).

    Args:
        cube: (H, W, B)
        gt:   (H, W)
        k:    number of bands
        penalty_lambda: redundancy penalty
        verbose: print logs

    Returns:
        selected_bands: list of band indices
        acc: accuracy on training set
        f1:  macro-F1 on training set
    """
    if verbose:
        print("Running SRPA...")

    H, W, B = cube.shape
    cube_flat = cube.reshape(-1, B)
    mask_flat = gt.flatten()
    valid = mask_flat > 0
    X = cube_flat[valid]
    y = mask_flat[valid]

    scores = srpa_scores(X, y, penalty_lambda=penalty_lambda)
    selected = np.argsort(scores)[::-1][:k]

    if verbose:
        print(f"Top-{k} bands (SRPA): {selected.tolist()}")

    X_sel = X[:, selected]
    rf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    rf2.fit(X_sel, y)
    y_pred = rf2.predict(X_sel)

    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))

    if verbose:
        print(f"[SRPA] Accuracy={acc:.4f}  F1={f1:.4f}")

    return [int(b) for b in selected], acc, f1