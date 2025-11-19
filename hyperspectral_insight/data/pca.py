import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def pca_reduce(
    
    cube: np.ndarray,
    gt: np.ndarray,
    var_thresh: float = 0.99,
    max_components: int = 30,
):
    """
    Fit PCA on labeled pixels -> return reduced cube.
    Used in full PCA experiments.
    """

    X, y = cube_to_Xy(cube, gt, mask_zero=True)

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    pca_full = PCA(n_components=min(max_components, Xz.shape[1]), whiten=False)
    pca_full.fit(Xz)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    D = int(np.searchsorted(cum, var_thresh) + 1)
    D = min(D, max_components)

    # Re-fit using D components
    pca = PCA(n_components=D).fit(Xz)
    Xp = pca.transform(Xz)

    # Place back in cube shape
    H, W, _ = cube.shape
    reduced = np.zeros((H * W, D), dtype=np.float32)
    mask = (gt.reshape(-1) != 0)
    reduced[mask] = Xp
    reduced_cube = reduced.reshape(H, W, D)

    return reduced_cube, scaler, pca, D

def pca_fit_transform_train(X: np.ndarray, var_thresh=0.99, max_components=30):
    """
    PCA for cross-validation:
    Fit on training spectra only.
    """
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    pca_full = PCA(n_components=min(max_components, Xz.shape[1]))
    pca_full.fit(Xz)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    D = int(np.searchsorted(cum, var_thresh) + 1)

    pca = PCA(n_components=D).fit(Xz)
    return scaler, pca, D

def pca_apply_to_patches(
    X_raw: np.ndarray, scaler: StandardScaler, pca: PCA, D: int
):
    """
    Apply PCA transform to patch tensors.
    """
    N, H, W, B, _ = X_raw.shape
    Xf = X_raw.reshape(-1, B)

    Xf = scaler.transform(Xf)
    Xp = pca.transform(Xf)[:, :D]

    return Xp.reshape(N, H, W, D, 1)