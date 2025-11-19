import numpy as np
from typing import Tuple

def cube_to_Xy(cube: np.ndarray, gt: np.ndarray, mask_zero=True):
    """
    Flatten (H,W,B) cube to (N,B) + labels.
    Used for PCA, IBRA, SRPA, etc.
    """
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    y = gt.reshape(-1)

    if mask_zero:
        mask = y != 0
        return X[mask], y[mask]

    return X, y

def extract_patches(
    cube: np.ndarray, gt: np.ndarray, win: int = 25, drop_label0: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (win x win) spatial patches around each labeled pixel.
    Output shape: (N, win, win, B, 1)
    """
    H, W, B = cube.shape
    pad = win // 2

    padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode="reflect")

    if drop_label0:
        ys, xs = np.where(gt != 0)
    else:
        ys, xs = np.where(np.ones_like(gt, dtype=bool))

    patches = []
    labels = []

    for r, c in zip(ys, xs):
        pr, pc = r + pad, c + pad
        patch = padded[pr-pad : pr+pad+1, pc-pad : pc+pad+1, :]
        patches.append(patch[..., None])
        labels.append(gt[r, c] - 1)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def create_patches(cube: np.ndarray, gt: np.ndarray, patch_size: int = 5):
    """
    Mini patch extraction for shallow 2D CNN baselines.
    Output: (N, patch, patch, B)
    """
    pad = patch_size // 2
    padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode="reflect")

    X, y = [], []
    for r in range(gt.shape[0]):
        for c in range(gt.shape[1]):
            if gt[r, c] == 0:
                continue
            patch = padded[r:r+patch_size, c:c+patch_size, :]
            X.append(patch)
            y.append(gt[r, c] - 1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def class_count(y: np.ndarray) -> int:
    return int(np.max(y) + 1)