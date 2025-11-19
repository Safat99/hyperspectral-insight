import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(data_file, gt_file, data_key, gt_key):
    data = sio.loadmat(data_file)[data_key]
    gt = sio.loadmat(gt_file)[gt_key]
    return data, gt



data, gt = load_dataset(
    "indian_pines/Indian_pines_corrected.mat",
    "indian_pines/Indian_pines_gt.mat",
    "indian_pines_corrected",
    "indian_pines_gt"
)

print("Data shape:", data.shape)   # (145, 145, 200)
print("GT shape:", gt.shape)
print(type(data))
print(type(gt))