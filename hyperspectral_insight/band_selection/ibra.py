import numpy as np
from tqdm import trange
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

def compute_vif_pair(band_i: np.ndarray, band_j: np.ndarray) -> float:
    """
    Compute Variance Inflation Factor between two bands (flattened).

    Args:
        band_i: (H, W) or (N,) array for band i.
        band_j: (H, W) or (N,) array for band j.

    Returns:
        VIF value (float).
    """
    x = band_j.reshape(-1, 1)
    y = band_i.reshape(-1)
    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)
    return float(1.0 / (1.0 - r2 + 1e-8))


def ibra_distances(
    cube: np.ndarray,
    threshold: float = 10.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IBRA-like method: compute left and right VIF-based distances per band.

    Args:
        cube: (H, W, D) hyperspectral cube.
        threshold: maximum VIF threshold (usually 10–12).
        verbose: show progress bar text if True.

    Returns:
        distances:      |left - right| per band (D,)
        distances_left: distance on left side per band (D,)
        distances_right: distance on right side per band (D,)
    """
    H, W, D = cube.shape
    vif_table = np.zeros((D, D), dtype=np.float32)
    distances_left = np.zeros(D, dtype=np.float32)
    distances_right = np.zeros(D, dtype=np.float32)

    iterator = trange(D, desc="Computing IBRA distances", disable=not verbose)

    for b in iterator:
        # Left direction
        d = 1
        vif = np.inf
        while vif > threshold and (b - d) >= 0:
            if vif_table[b, b - d] == 0:
                vif_table[b, b - d] = compute_vif_pair(
                    cube[..., b], cube[..., b - d]
                )
                vif_table[b - d, b] = vif_table[b, b - d]
            vif = vif_table[b, b - d]
            d += 1
        distances_left[b] = d - 1

        # Right direction
        d = 1
        vif = np.inf
        while vif > threshold and (b + d) < D:
            if vif_table[b, b + d] == 0:
                vif_table[b, b + d] = compute_vif_pair(
                    cube[..., b], cube[..., b + d]
                )
                vif_table[b + d, b] = vif_table[b, b + d]
            vif = vif_table[b, b + d]
            d += 1
        distances_right[b] = d - 1

    distances = np.abs(distances_left - distances_right)
    return distances, distances_left, distances_right


def ibra_band_candidates(
    distances: np.ndarray,
    max_distance: float = 5.0,
) -> List[int]:
    """
    Given IBRA distances, pick candidate band 'centers' as local minima.

    Args:
        distances: (D,) array from ibra_distances().
        max_distance: keep only bands whose distance < max_distance.

    Returns:
        List of candidate band indices.
    """
    from scipy.signal import find_peaks

    # pad with large values to detect valleys in the middle
    dist = np.concatenate(([100.0], distances, [100.0]))
    # peaks in inverted distances == valleys in original
    peaks, _ = find_peaks(np.max(dist) - dist)
    peaks = peaks - 1  # shift back because of padding

    candidates = [int(p) for p in peaks if distances[p] < max_distance]
    return candidates

