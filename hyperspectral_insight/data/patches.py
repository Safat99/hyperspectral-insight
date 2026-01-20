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

# def extract_patches(
#     cube: np.ndarray, gt: np.ndarray, win: int = 25, drop_label0: bool = True
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Extract (win x win) spatial patches around each labeled pixel.
#     Output shape: (N, win, win, B, 1)
#     """
#     H, W, B = cube.shape
#     pad = win // 2

#     padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode="reflect")

#     if drop_label0:
#         ys, xs = np.where(gt != 0)
#     else:
#         ys, xs = np.where(np.ones_like(gt, dtype=bool))

#     patches = []
#     labels = []

#     for r, c in zip(ys, xs):
#         pr, pc = r + pad, c + pad
#         patch = padded[pr-pad : pr+pad+1, pc-pad : pc+pad+1, :]
#         patches.append(patch[..., None])
#         labels.append(gt[r, c] - 1)

#     return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def extract_patches(
    cube: np.ndarray,
    gt: np.ndarray,
    win: int = 25,
    stride: int = 1,
    drop_label0: bool = True,
    max_samples_per_class: int = None
) -> Tuple[np.ndarray, np.ndarray]:

    H, W, B = cube.shape
    pad = win // 2

    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    if drop_label0:
        ys, xs = np.where(gt != 0)
    else:
        ys, xs = np.where(np.ones_like(gt, dtype=bool))

    coords = np.stack([ys, xs], axis=1)
    
    if stride > 1:
        coords = coords[
            (coords[:, 0] % stride == 0) &
            (coords[:, 1] % stride == 0)
        ]

    # ----- NEW: class-wise sampling -----
    if max_samples_per_class is not None:
        new_coords = []
        for cls in np.unique(gt):
            if cls == 0 and drop_label0:
                continue
            cls_mask = (gt[ys, xs] == cls)
            cls_coords = coords[cls_mask]

            if len(cls_coords) > max_samples_per_class:
                idx = np.random.choice(
                    len(cls_coords),
                    max_samples_per_class,
                    replace=False
                )
                cls_coords = cls_coords[idx]

            new_coords.append(cls_coords)

        coords = np.vstack(new_coords)

    # Extract patches
    patches = []
    labels = []

    for r, c in coords:
        pr, pc = r + pad, c + pad
        patch = padded[pr-pad: pr+pad+1, pc-pad: pc+pad+1, :]
        patches.append(patch[..., None])   # keep 3D CNN shape
        labels.append(gt[r, c] - 1)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def create_patches(cube: np.ndarray, gt: np.ndarray, patch_size: int = 5, max_samples_per_class: int = None):
    """
    Mini patch extraction for shallow 2D CNN baselines.
    Output: (N, patch, patch, B)
    
    If max_samples_per_class is set, randomly samples that many
    patches per class, preventing memory explosion (needed for Pavia Centre).
    """
    
    if max_samples_per_class is not None:
        raise ValueError(
            "create_patches(): max_samples_per_class is deprecated. "
            "Sampling must be performed inside cross-validation."
        )
    
    pad = patch_size // 2
    padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode="reflect")

    X, y = [], []
    
    for cls in np.unique(gt):
        if cls == 0:
            continue
    
        coords = np.argwhere(gt == cls)
        
        # if max_samples_per_class is not None and len(coords) > max_samples_per_class:
        #     idx = np.random.choice(len(coords), max_samples_per_class, replace=False)
        #     coords = coords[idx]
        
        for r, c in coords:
            patch = padded[r:r+patch_size, c:c+patch_size, :]
            X.append(patch)
            y.append(cls - 1)
    
    # for r in range(gt.shape[0]):
    #     for c in range(gt.shape[1]):
    #         if gt[r, c] == 0:
    #             continue
    #         patch = padded[r:r+patch_size, c:c+patch_size, :]
    #         X.append(patch)
    #         y.append(gt[r, c] - 1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def class_count(y: np.ndarray) -> int:
    return int(np.max(y) + 1)