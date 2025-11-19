import scipy.io as sio
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


# if we want to keep aside any private method not to import.
# We can then exclude it from this __all__ variable
__all__ = [
    "load_mat_dataset",
    "load_dataset_from_paths",
    "DATASETS",
    "load_dataset"
]


# Dataset registry

DATASETS: Dict[str, Dict[str, str]] = {
    "indian_pines": {
        "data": "datasets/indian_pines/Indian_pines_corrected.mat",
        "gt":   "datasets/indian_pines/Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key":   "indian_pines_gt",
    },
    "salinas": {
        "data": "datasets/salinas/Salinas_corrected.mat",
        "gt":   "datasets/salinas/Salinas_gt.mat",
        "data_key": "salinas_corrected",
        "gt_key":   "salinas_gt",
    },
    "pavia_university": {
        "data": "datasets/pavia/university/PaviaU.mat",
        "gt":   "datasets/pavia/university/PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key":   "paviaU_gt",
    },
    "pavia_centre": {
        "data": "datasets/pavia/centre/Pavia.mat",
        "gt":   "datasets/pavia/centre/Pavia_gt.mat",
        "data_key": "pavia",
        "gt_key":   "pavia_gt",
    },
    "ksc": {
        "data": "datasets/ksc/KSC_corrected.mat",
        "gt":   "datasets/ksc/KSC_gt.mat",
        "data_key": "KSC",
        "gt_key":   "KSC_gt",
    }
}

# Load .mat files


def load_mat_dataset(
    data_file: str, gt_file: str, data_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hyperspectral data & ground truth from .mat files.
    Returns:
        cube: (H, W, B)
        gt:   (H, W)
    """
    data = sio.loadmat(data_file)[data_key]
    gt = sio.loadmat(gt_file)[gt_key]

    data = data.astype(np.float32)
    gt = gt.astype(np.int32)

    return data, gt

def load_dataset_from_paths(
    data_path: str, gt_path: str, data_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around load_mat_dataset with proper path handling."""
    data_path = Path(data_path)
    gt_path = Path(gt_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    return load_mat_dataset(str(data_path), str(gt_path), data_key, gt_key)



# Top-level loader which will be used in all experiements
def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset by name using the registry.
    Example:
        cube, gt = load_dataset("indian_pines")
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]
    return load_dataset_from_paths(cfg["data"], cfg["gt"], cfg["data_key"], cfg["gt_key"])

