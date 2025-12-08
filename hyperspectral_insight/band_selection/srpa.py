import numpy as np
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

import tensorflow as tf
from tensorflow.keras import layers, models

def srpa_scores(
    X: np.ndarray,
    y: np.ndarray,
    penalty_lambda: float = 0.3,
) -> np.ndarray:
    """
    Compute SRPA scores for each band.

    Args:
        X: (N, B) spectral samples.
        y: (N,) labels.
        penalty_lambda: redundancy penalty weight.

    Returns:
        srpa_score: (B,) array of scores (higher is better).
    """
    B = X.shape[1]

    # RF importances (attention weights)
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X, y)
    attention = rf.feature_importances_

    # Redundancy via correlation
    corr = np.zeros(B, dtype=np.float32)
    for i in range(B):
        vals = []
        for j in range(B):
            if j == i:
                continue
            r, _ = pearsonr(X[:, i], X[:, j])
            vals.append(abs(r))
        corr[i] = np.mean(vals) if vals else 0.0

    srpa_score = attention - penalty_lambda * corr
    return srpa_score


def select_bands_srpa(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
    penalty_lambda: float = 0.3,
) -> List[int]:
    """
    Select bands via SRPA from an image cube.

    Args:
        cube: (H, W, B)
        gt:   (H, W)
        k:    number of bands
        penalty_lambda: redundancy penalty

    Returns:
        List of selected band indices.
    """
    if cube.ndim == 3:
        H, W, B = cube.shape
        cube_flat = cube.reshape(-1, B)
        mask_flat = gt.flatten()
        valid = mask_flat > 0
        X = cube_flat[valid]
        y = mask_flat[valid]
    else:
        raise ValueError("cube must be 3D (H,W,B) for select_bands_srpa")

    scores = srpa_scores(X, y, penalty_lambda=penalty_lambda)
    selected = np.argsort(scores)[::-1][:k]
    return [int(b) for b in selected]


def run_srpa_pipeline(
    cube: np.ndarray,
    gt: np.ndarray,
    k: int = 20,
    penalty_lambda: float = 0.3,
    verbose: bool = True,
) -> Tuple[List[int], float, float]:
    """
    Run SRPA + quick RF evaluation (on selected bands).

    Args:
        cube: (H, W, B)
        gt:   (H, W)
        k:    number of bands
        penalty_lambda: redundancy penalty
        verbose: print logs

    Returns:
        selected_bands: list of band indices
        acc: accuracy on training set
        f1:  macro-F1 on training set
    """
    if verbose:
        print("Running SRPA...")

    H, W, B = cube.shape
    cube_flat = cube.reshape(-1, B)
    mask_flat = gt.flatten()
    valid = mask_flat > 0
    X = cube_flat[valid]
    y = mask_flat[valid]

    scores = srpa_scores(X, y, penalty_lambda=penalty_lambda)
    selected = np.argsort(scores)[::-1][:k]

    if verbose:
        print(f"Top-{k} bands (SRPA): {selected.tolist()}")

    X_sel = X[:, selected]
    rf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    rf2.fit(X_sel, y)
    y_pred = rf2.predict(X_sel)

    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))

    if verbose:
        print(f"[SRPA] Accuracy={acc:.4f}  F1={f1:.4f}")

    return [int(b) for b in selected], acc, f1


################ 3dcnn spra mechanism ##########################

class SpatialMean(layers.Layer):
    """Average over H and W dimensions only."""
    def call(self, x):
        return tf.reduce_mean(x, axis=[1, 2])  # (B, bands, C)


def build_srpa_3dcnn(num_bands, num_classes):
    """
    Input : (25, 25, B, 1)
    Output: (class_logits, attention_weights[B])
    """

    inp = layers.Input(shape=(25, 25, num_bands, 1))

    # ---------------------------- 3D CNN FRONT ----------------------------
    x = layers.Conv3D(8, (3,3,3), padding="same", activation="relu")(inp)
    x = layers.MaxPool3D(pool_size=(2,2,1))(x)

    x = layers.Conv3D(16, (3,3,3), padding="same", activation="relu")(x)
    # x shape is now (batch, H/2, W/2, B, 16)

    # ------------------- GLOBAL SPATIAL AVG (KEEP BANDS) -------------------
    # result shape: (batch, B, 16)
    x = SpatialMean()(x)

    # ------------------------ SE ATTENTION BLOCK ---------------------------
    # x: (batch, B, 16)
    band_feat = layers.GlobalAveragePooling1D()(x)  # (batch, B)
    
    r = max(1, num_bands // 4)
    se = layers.Dense(r, activation="relu")(band_feat)
    se = layers.Dense(num_bands, activation="sigmoid")(se)

    # --------------------------- CLASSIFIER --------------------------------
    flat = layers.Flatten()(x)
    out = layers.Dense(num_classes, activation="softmax")(flat)

    model = models.Model(inp, [out, se])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=["categorical_crossentropy", None],
        metrics=[["accuracy"], None]
    )
    return model



# ---------------------------------------------------------
# Full SRPA selection
# ---------------------------------------------------------

def srpa_selection_3dcnn(
    cube, gt,
    patch_size=25,
    num_bands=20,
    epochs=2,
    batch_size=32,
    lambda_penalty=0.3,
):
    H, W, B = cube.shape

    # -------------------- Extract Patches --------------------
    patches = []
    labels = []
    pad = patch_size // 2

    padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode="reflect")
    gt_pad = np.pad(gt, ((pad,pad),(pad,pad)), mode="reflect")

    for i in range(H):
        for j in range(W):
            if gt[i,j] == 0: continue
            patches.append(padded[i:i+patch_size, j:j+patch_size, :])
            labels.append(gt[i,j])

    patches = np.array(patches)          # (N,25,25,B)
    labels = np.array(labels)

    # Convert labels to 1..C → 0..C-1 → one-hot
    labels -= labels.min()
    num_classes = labels.max() + 1
    y_oh = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    X = patches[..., np.newaxis]         # (N,25,25,B,1)

    # -------------------- Build SRPA Model --------------------
    model = build_srpa_3dcnn(num_bands=B, num_classes=num_classes)

    # -------------------- Train -------------------------
    model.fit(
        X, [y_oh, np.zeros((len(X), B))],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # -------------------- Collect Attention ----------------
    _, attn = model.predict(X, batch_size=batch_size, verbose=0)
    attn_mean = attn.mean(axis=0)

    # -------------------- Compute Redundancy ---------------
    X_flat = cube.reshape(-1, B)
    corr = np.corrcoef(X_flat, rowvar=False)
    redundancy = (np.abs(corr).sum(axis=1) - 1) / (B - 1)

    # -------------------- Final SRPA score -----------------
    srpa_score = attn_mean - lambda_penalty * redundancy
    topk = np.argsort(srpa_score)[::-1][:num_bands]

    return topk.tolist(), srpa_score

