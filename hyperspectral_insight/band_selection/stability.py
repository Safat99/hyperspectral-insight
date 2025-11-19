import numpy as np
from typing import List, Tuple

from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def to_binary_matrix(fold_band_lists: List[List[int]], n_bands: int) -> np.ndarray:
    """
    Convert list-of-bands per fold to a binary fold×bands matrix.

    Args:
        fold_band_lists: list of length F, each a list of band indices.
        n_bands: total available bands.

    Returns:
        M: (F, n_bands) binary matrix.
    """
    F = len(fold_band_lists)
    M = np.zeros((F, n_bands), dtype=int)
    for i, bands in enumerate(fold_band_lists):
        M[i, np.array(bands, dtype=int)] = 1
    return M


def hamming_distance_matrix(M: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Hamming distance matrix between rows of M.

    Args:
        M: (F, n_bands) binary matrix.

    Returns:
        D: (F, F) Hamming distance matrix.
    """
    D = squareform(pdist(M, metric="hamming"))
    return D


def similarity_from_hamming(D: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """
    Convert Hamming distances into similarities via exp(-λ d²).

    Args:
        D: (F, F) distance matrix.
        lam: lambda for exponential kernel.

    Returns:
        S: (F, F) similarity matrix.
    """
    S = np.exp(-lam * (D ** 2))
    np.fill_diagonal(S, 1.0)
    return S


def plot_similarity(S: np.ndarray, title: str = "Band-Selection Similarity") -> None:
    """
    Visualize similarity matrix.

    Args:
        S: (F, F) similarity matrix.
        title: plot title.
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(S, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Similarity")
    plt.title(title)
    plt.xlabel("Fold j")
    plt.ylabel("Fold i")
    plt.tight_layout()
    plt.show()


def compute_band_selection_stability(
    fold_band_lists: List[List[int]],
    n_bands: int,
    lam: float = 10.0,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stability of band selection across folds.

    Args:
        fold_band_lists: list of per-fold selected bands.
        n_bands: total band count.
        lam: similarity kernel parameter.
        plot: whether to plot similarity matrix.

    Returns:
        S: similarity matrix (F, F)
        D: Hamming distance matrix (F, F)
        M: binary matrix (F, n_bands)
    """
    M = to_binary_matrix(fold_band_lists, n_bands)
    D = hamming_distance_matrix(M)
    S = similarity_from_hamming(D, lam=lam)

    if plot:
        plot_similarity(S, title="exp(-λ Hamming²) Band-Selection Similarity")

    # summary stats
    off_diag = S[np.triu_indices_from(S, k=1)]
    print(f"Mean similarity = {off_diag.mean():.4f}")
    print(f"Std similarity  = {off_diag.std():.4f}")

    return S, D, M