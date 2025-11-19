import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Optional

from hyperspectral_insight.ssl.ssl_utils import merge_labeled_and_pseudo

def predict_proba_in_batches(model, X, batch_size=32):
    """
    Run model.predict on X in batches, return probabilities.

    Works with any Keras model that outputs class probabilities.
    """
    n = X.shape[0]
    probs_list = []
    for i in range(0, n, batch_size):
        batch = X[i : i + batch_size]
        p = model.predict(batch, verbose=0)
        probs_list.append(p)
    return np.vstack(probs_list)

def generate_pseudo_labels(
    model,
    X_u: np.ndarray,
    confidence_threshold: float = 0.9,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate pseudo-labels from an unlabeled pool using a trained model.

    Args:
        model: Keras model with .predict(), outputs probabilities.
        X_u: (N_u, ...) unlabeled samples.
        confidence_threshold: minimum max-prob to accept a pseudo-label.
        batch_size: prediction batch size.

    Returns:
        X_pseudo: samples accepted as pseudo-labeled
        y_pseudo: integer pseudo-labels
        confidences: max probability for each pseudo-labeled sample
        selected_idx: indices in X_u corresponding to pseudo-labeled samples
    """
    if X_u.shape[0] == 0:
        return (
            np.empty((0,) + X_u.shape[1:], dtype=X_u.dtype),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    probs = predict_proba_in_batches(model, X_u, batch_size=batch_size)
    max_conf = probs.max(axis=1)
    y_hat = probs.argmax(axis=1)

    mask = max_conf >= confidence_threshold
    selected_idx = np.where(mask)[0]

    X_pseudo = X_u[selected_idx]
    y_pseudo = y_hat[selected_idx]
    confidences = max_conf[selected_idx]

    return X_pseudo, y_pseudo, confidences, selected_idx


def iterative_pseudo_labeling(
    model_fn,
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: Optional[int] = None,
    n_iters: int = 5,
    confidence_threshold: float = 0.9,
    max_pseudo_per_iter: Optional[int] = None,
    batch_size: int = 32,
    epochs_per_iter: int = 20,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, List[Dict]]:
    """
    Simple iterative SSL training loop using pseudo-labeling.

    At each iteration:
        1) Train model on current labeled + pseudo-labeled set
        2) Evaluate on validation set
        3) Generate pseudo-labels on remaining unlabeled pool
        4) Add newly pseudo-labeled samples to training set
        5) Remove them from unlabeled pool

    Args:
        model_fn: function (input_shape, n_classes) -> compiled Keras model
        X_l, y_l: initial labeled data
        X_u: initial unlabeled data
        X_val, y_val: validation set
        n_classes: number of classes (if None, inferred from y_l)
        n_iters: number of pseudo-labeling iterations
        confidence_threshold: minimum probability to accept pseudo-label
        max_pseudo_per_iter: optional cap on new pseudo-labels each iteration
        batch_size: training and prediction batch size
        epochs_per_iter: Keras epochs per iteration
        verbose: passed to model.fit()

    Returns:
        model: final trained model
        history: list of dicts with metrics per iteration
    """
    if n_classes is None:
        n_classes = int(y_l.max() + 1)

    # One-hot encode labels
    y_l_oh = tf.keras.utils.to_categorical(y_l, num_classes=n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    input_shape = X_l.shape[1:]
    model = model_fn(input_shape, n_classes=n_classes)

    history = []

    X_train = X_l.copy()
    y_train_oh = y_l_oh.copy()
    X_unlabeled = X_u.copy()

    for it in range(n_iters):
        print(f"\n[SSL] Iteration {it + 1}/{n_iters}")
        print(f"  Labeled size: {X_train.shape[0]}  |  Unlabeled size: {X_unlabeled.shape[0]}")

        # Train / fine-tune model on current labeled + pseudo data
        hist = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs_per_iter,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Evaluate on validation
        val_probs = model.predict(X_val, verbose=0)
        y_val_pred = val_probs.argmax(axis=1)
        val_acc = (y_val_pred == y_val).mean()

        # Generate pseudo labels on current unlabeled pool
        X_pseudo, y_pseudo, confidences, selected_idx = generate_pseudo_labels(
            model,
            X_unlabeled,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
        )

        n_new = X_pseudo.shape[0]

        if max_pseudo_per_iter is not None and n_new > max_pseudo_per_iter:
            # select top-k by confidence
            order = np.argsort(-confidences)
            topk = order[:max_pseudo_per_iter]
            X_pseudo = X_pseudo[topk]
            y_pseudo = y_pseudo[topk]
            selected_idx = selected_idx[topk]
            n_new = X_pseudo.shape[0]

        print(f"  New pseudo-labels accepted: {n_new}")

        # Record iteration info
        iter_info = {
            "iteration": it,
            "train_size": int(X_train.shape[0]),
            "unlabeled_size": int(X_unlabeled.shape[0]),
            "n_new_pseudo": int(n_new),
            "val_acc": float(val_acc),
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        }
        history.append(iter_info)

        # If no new pseudo-labels, stop
        if n_new == 0:
            print("  No new pseudo-labels added; stopping early.")
            break

        # Merge new pseudo-labeled data into training set
        X_train, y_train_oh = merge_labeled_and_pseudo(
            X_train,
            y_train_oh,
            X_pseudo,
            y_pseudo,
            n_classes=n_classes,
        )

        # Remove pseudo-labeled samples from unlabeled pool
        mask_keep = np.ones(X_unlabeled.shape[0], dtype=bool)
        mask_keep[selected_idx] = False
        X_unlabeled = X_unlabeled[mask_keep]

        if X_unlabeled.shape[0] == 0:
            print("  Unlabeled pool exhausted; stopping.")
            break

    return model, history