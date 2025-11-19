import numpy as np
from typing import List, Optional


from .ibra import ibra_distances, ibra_band_candidates
from .gss import gss_selection

def select_bands_ibra_gss(
    cube: np.ndarray,
    nbands: int = 5,
    vif_threshold: float = 10.0,
    max_distance: float = 5.0,
    verbose: bool = False,
) -> List[int]:
    """
    Full IBRA+GSS band selection pipeline on an HSI cube.

    Args:
        cube: (H, W, D)
        nbands: number of final bands to pick.
        vif_threshold: IBRA VIF threshold.
        max_distance: pruning threshold for IBRA distances.
        verbose: print debug info if True.

    Returns:
        List of selected band indices.
    """
    distances, _, _ = ibra_distances(cube, threshold=vif_threshold, verbose=verbose)
    ibra_bands = ibra_band_candidates(distances, max_distance=max_distance)

    if verbose:
        print(f"[IBRA] Candidates: {ibra_bands}")

    selected = gss_selection(cube, ibra_bands, nbands=nbands)
    if verbose:
        print(f"[IBRA+GSS] Selected bands: {selected}")

    return selected

def select_bands_ibra_gss_on_patches(
    train_patches: np.ndarray,
    nbands: int = 5,
    vif_threshold: float = 10.0,
    max_distance: float = 5.0,
    verbose: bool = False,
) -> List[int]:
    """
    Apply IBRA+GSS on a 'pseudo cube' reconstructed from training patches.

    This approximates image-level redundancy using patch-level data.

    Args:
        train_patches: (N, H, W, D, 1)
        nbands: number of bands to pick.
        vif_threshold: IBRA VIF threshold.
        max_distance: pruning threshold for IBRA distances.
        verbose: print debug info if True.

    Returns:
        List of selected band indices.
    """
    N, H, W, D, _ = train_patches.shape
    # collapse patches back into pseudo-cube: (H', W', D)
    pseudo_flat = train_patches[..., 0].reshape(-1, D)
    n_pix = pseudo_flat.shape[0]
    side = int(np.sqrt(n_pix))

    pseudo_cube = pseudo_flat[: side * side, :].reshape(side, side, D)

    return select_bands_ibra_gss(
        pseudo_cube,
        nbands=nbands,
        vif_threshold=vif_threshold,
        max_distance=max_distance,
        verbose=verbose,
    )
    
class BandSelectionPipeline:
    """
    Lightweight pipeline wrapper for band selection.

    Currently supports:
        - method="ibra_gss": fully unsupervised IBRA+GSS on cube
        (SSEP / SRPA remain as separate supervised functions.)
    """

    def __init__(
        self,
        method: str = "ibra_gss",
        nbands: int = 5,
        vif_threshold: float = 10.0,
        max_distance: float = 5.0,
        verbose: bool = False,
    ):
        self.method = method.lower()
        self.nbands = nbands
        self.vif_threshold = vif_threshold
        self.max_distance = max_distance
        self.verbose = verbose

        if self.method not in {"ibra_gss"}:
            raise ValueError(
                f"Unsupported method '{method}'. Currently only 'ibra_gss' is implemented in BandSelectionPipeline."
            )

    def select(
        self,
        cube: Optional[np.ndarray] = None,
        train_patches: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Run the pipeline.

        For method="ibra_gss":
            - if cube is provided, selection is done on the cube
            - if train_patches is provided, we use the pseudo-cube trick

        Returns:
            List of selected band indices.
        """
        if self.method == "ibra_gss":
            if cube is not None:
                return select_bands_ibra_gss(
                    cube,
                    nbands=self.nbands,
                    vif_threshold=self.vif_threshold,
                    max_distance=self.max_distance,
                    verbose=self.verbose,
                )
            elif train_patches is not None:
                return select_bands_ibra_gss_on_patches(
                    train_patches,
                    nbands=self.nbands,
                    vif_threshold=self.vif_threshold,
                    max_distance=self.max_distance,
                    verbose=self.verbose,
                )
            else:
                raise ValueError(
                    "For method 'ibra_gss', either 'cube' or 'train_patches' must be provided."
                )

        raise ValueError(f"Unsupported method '{self.method}' in BandSelectionPipeline.")
