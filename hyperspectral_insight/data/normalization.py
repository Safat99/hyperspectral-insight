import numpy as np
from typing import Tuple

def minmax_normalize(cube: np.ndarray) -> np.ndarray:
    """
    Normalize per-band to [0,1].
    Safe for CNNs and 3D CNNs.
    """
    cube = cube.astype(np.float32)
    cmin = np.min(cube, axis=(0,1), keepdims=True)
    cmax = np.max(cube, axis=(0,1), keepdims=True)
    return (cube - cmin) / (cmax - cmin + 1e-8)

def zscore_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize spectra: (N,B) but can be used with patches.
    Returns X_norm, mean, std.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std