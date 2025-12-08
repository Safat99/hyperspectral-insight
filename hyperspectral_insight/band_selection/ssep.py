import numpy as np
from typing import List, Tuple

from skimage.filters import sobel
from skimage import img_as_float
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def ssep_scores(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute SSEP Dice-based edge alignment scores per band.

    Args:
        cube: (H, W, B)
        gt:   (H, W) ground truth labels

    Returns:
        dice_scores: (B,) array with per-band scores
    """
    H, W, B = cube.shape
    cube = img_as_float(cube)
    mask = gt.astype(int)

    # label edges (binary)
    mask_edges = sobel(mask > 0) > 0

    dice_scores = np.zeros(B, dtype=np.float32)

    for i in range(B):
        band_img = cube[:, :, i]
        band_edges = sobel(band_img) > 0

        intersection = np.logical_and(mask_edges, band_edges).sum()
        denom = mask_edges.sum() + band_edges.sum() + 1e-8
        dice_scores[i] = 2.0 * intersection / denom

    return dice_scores


def select_bands_ssep(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
) -> List[int]:
    """
    Select top-k bands using SSEP scores.

    Args:
        cube: (H, W, B)
        gt:   (H, W)
        k:    number of bands.

    Returns:
        List of selected band indices.
    """
    scores = ssep_scores(cube, gt)
    selected = np.argsort(scores)[::-1][:k]
    return [int(b) for b in selected]


# def run_ssep_pipeline(
#     cube: np.ndarray,
#     gt: np.ndarray,
#     k: int = 20,
#     model_type: str = "rf",
#     verbose: bool = True,
# ) -> Tuple[List[int], float, float]:
#     """
#     Full SSEP pipeline + quick RF/SVM evaluation.

#     Args:
#         cube: (H, W, B)
#         gt:   (H, W)
#         k:    number of bands.
#         model_type: 'rf' or 'svm'
#         verbose: print logs

#     Returns:
#         selected_bands: list of selected band indices
#         acc: training-set accuracy of quick classifier
#         f1:  macro-F1 of quick classifier
#     """
#     if verbose:
#         print("Running SSEP...")

#     H, W, B = cube.shape
#     cube = img_as_float(cube)
#     mask = gt.astype(int)

#     selected_bands = select_bands_ssep(cube, mask, k=k)
#     if verbose:
#         print(f"Top-{k} bands (SSEP): {selected_bands}")

#     cube_flat = cube.reshape(-1, B)
#     mask_flat = mask.flatten()
#     valid = mask_flat > 0
#     X = cube_flat[valid][:, selected_bands]
#     y = mask_flat[valid]

#     if model_type == "rf":
#         clf = RandomForestClassifier(n_estimators=100, random_state=0)
#     else:
#         from sklearn.svm import SVC
#         clf = SVC(kernel="rbf")

#     clf.fit(X, y)
#     y_pred = clf.predict(X)

#     acc = float(accuracy_score(y, y_pred))
#     f1 = float(f1_score(y, y_pred, average="macro"))

#     if verbose:
#         print(f"[SSEP] Accuracy={acc:.4f}  F1={f1:.4f}")

#     return selected_bands, acc, f1

def run_ssep_pipeline(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
    verbose: bool = True,
):
    """
    SSEP band-selection
    Returns only the selected band indices.
    """
    if verbose:
        print("Running SSEP (band-selection only)...")

    cube = img_as_float(cube)
    mask = gt.astype(int)

    # compute top-k bands
    selected_bands = select_bands_ssep(cube, mask, k=k)

    if verbose:
        print(f"Selected top-{k} SSEP bands: {selected_bands}")

    # return ONLY band indices
    return selected_bands