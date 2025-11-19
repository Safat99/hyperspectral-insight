import numpy as np
from typing import List, Sequence, Tuple

def band_entropy(cube: np.ndarray, band_idx: int, bins: int = 100) -> float:
    """
    Compute Shannon entropy of a single band.

    Args:
        cube: (H, W, D) hyperspectral cube.
        band_idx: band index to analyze.
        bins: histogram bins.

    Returns:
        Entropy (float).
    """
    band = cube[..., band_idx].ravel()
    hist, _ = np.histogram(band, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def gss_selection(
    cube: np.ndarray,
    ibra_bands: Sequence[int],
    nbands: int = 5,
    bins: int = 100,
) -> List[int]:
    """
    GSS-style selection: from IBRA candidates, pick bands with highest entropy.

    Args:
        cube: (H, W, D) hyperspectral cube.
        ibra_bands: iterable of candidate band indices (from IBRA).
        nbands: number of bands to select.
        bins: histogram bins for entropy.

    Returns:
        List of selected band indices (length <= nbands).
    """
    entropies: List[Tuple[int, float]] = []
    for b in ibra_bands:
        e = band_entropy(cube, b, bins=bins)
        entropies.append((int(b), float(e)))

    entropies.sort(key=lambda x: x[1], reverse=True)
    selected = [b for b, _ in entropies[:nbands]]
    return selected