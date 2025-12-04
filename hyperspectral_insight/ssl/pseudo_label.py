# hyperspectral_insight/ssl/pseudo_label.py

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Optional

from hyperspectral_insight.ssl.ssl_utils import merge_labeled_and_pseudo


def predict_proba_in_batches(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run model.predict on X in batches to avoid memory spikes.
    Assumes model outputs class probabilities (softmax).
    """
    if X is None or X.shape[0] == 0:
        return np.empty((0,))

    n = X.shape[0]
    probs_list = []
    for i in range(0, n, batch_size):
        batch = X[i: i + batch_size]
        p = model.predict(batch, verbose=0)
        probs_list.append(p)
    return np.vstack(probs_list)


def generate_pseudo_labels(
    model: tf.keras.Model,
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
    if X_u is None or X_u.shape[0] == 0:
        # Return empty arrays with consistent shapes/dtypes
        return (
            np.empty((0,) + (X_u.shape[1:] if X_u is not None else ()), dtype=X_u.dtype if X_u is not None else np.float32),
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
    X_u: Optional[np.ndarray],
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
    Semi-supervised iterative pseudo-labeling loop for ONE fold.

    At each iteration:
        1) Train / fine-tune model on current labeled + pseudo-labeled set
        2) Evaluate on validation set
        3) Generate pseudo-labels on remaining unlabeled pool
        4) Add newly pseudo-labeled samples to training set
        5) Remove them from unlabeled pool

    If X_u is None, or n_iters <= 0, or confidence_threshold is None,
    this degenerates to supervised-only training on (X_l, y_l).

    Args:
        model_fn: function (input_shape, n_classes) -> compiled Keras model
        X_l, y_l: initial labeled data for this fold
        X_u: unlabeled pool (or None)
        X_val, y_val: validation set
        n_classes: number of classes (if None, inferred from y_l)
        n_iters: number of pseudo-labeling iterations
        confidence_threshold: minimum probability to accept pseudo-label
        max_pseudo_per_iter: cap on new pseudo-labels per iteration
        batch_size: training and prediction batch size
        epochs_per_iter: Keras epochs per iteration
        verbose: passed to model.fit()

    Returns:
        model: final trained model
        history: list of dicts with metrics per iteration
    """
    if n_classes is None:
        n_classes = int(y_l.max() + 1)

    input_shape = X_l.shape[1:]

    # One-hot encode labels
    y_l_oh = tf.keras.utils.to_categorical(y_l, num_classes=n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    # Decide if we actually run SSL or just supervised
    run_ssl = (
        X_u is not None
        and X_u.shape[0] > 0
        and n_iters > 0
        and confidence_threshold is not None
    )

    # Initialize model once; we will continue training across iterations
    model = model_fn(input_shape, n_classes=n_classes)

    # Local copies so we don't modify caller arrays
    X_train = X_l.copy()
    y_train_oh = y_l_oh.copy()
    X_unlabeled = X_u.copy() if (X_u is not None and X_u.shape[0] > 0) else np.empty((0,) + X_l.shape[1:], dtype=X_l.dtype)

    history: List[Dict] = []

    if not run_ssl:
        # -----------------------------
        # Supervised-only training
        # -----------------------------
        print("[iterative_pseudo_labeling] No valid unlabeled pool or SSL config → supervised-only training.")

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

        iter_info = {
            "iteration": 0,
            "train_size": int(X_train.shape[0]),
            "unlabeled_size": int(X_unlabeled.shape[0]),
            "n_new_pseudo": 0,
            "val_acc": float(val_acc),
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        }
        history.append(iter_info)

        return model, history

    # -----------------------------
    # SSL pseudo-label iterations
    # -----------------------------
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

        if X_unlabeled.shape[0] == 0:
            print("  Unlabeled pool exhausted; stopping.")
            iter_info = {
                "iteration": it,
                "train_size": int(X_train.shape[0]),
                "unlabeled_size": 0,
                "n_new_pseudo": 0,
                "val_acc": float(val_acc),
                "last_epoch_train_loss": float(hist.history["loss"][-1]),
                "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
            }
            history.append(iter_info)
            break

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


def run_pseudo_label_ssl_fold(
    build_model_fn,
    X_labeled_fold: np.ndarray,
    y_labeled_fold: np.ndarray,
    X_unlabeled_pool: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs_per_iter: int = 20,
    batch_size: int = 16,
    confidence_thresh: Optional[float] = 0.9,
    ssl_iters: int = 5,
    val_frac: float = 0.2,
) -> Dict:
    """
    High-level wrapper for a SINGLE FOLD:

        - split fold-labeled into train/val (e.g. 80/20)
        - if unlabeled data is provided and ssl_iters > 0 and confidence_thresh is not None:
            run iterative pseudo-labeling with the unlabeled pool
        - otherwise:
            run supervised-only baseline
        - evaluate final model on TEST set

    Args:
        build_model_fn: (input_shape, n_classes) -> compiled model
        X_labeled_fold, y_labeled_fold: fold's labeled pool
        X_unlabeled_pool: full unlabeled pool (or None for baseline)
        X_test, y_test: held-out test set
        epochs_per_iter, batch_size, confidence_thresh, ssl_iters: SSL params
        val_frac: fraction of fold labeled data used as validation.
                  If 0.0, we use X_test/y_test as validation instead.

    Returns:
        dict with:
            - OA_test: float
            - history: list of per-iteration dicts
    """
    N = X_labeled_fold.shape[0]
    n_classes = int(y_labeled_fold.max() + 1)

    # Shuffle within the fold for robustness
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    X_l = X_labeled_fold[perm]
    y_l = y_labeled_fold[perm]

    # ------------------------------------------
    # Build train/val split
    # ------------------------------------------
    if val_frac > 0.0 and N > 1:
        split = int((1.0 - val_frac) * N)
        # ensure both train and val are non-empty
        split = max(1, min(split, N - 1))
        X_train_l = X_l[:split]
        y_train_l = y_l[:split]
        X_val = X_l[split:]
        y_val = y_l[split:]
    else:
        # No internal val split requested:
        # use all labeled as train, and TEST set as validation.
        # (In your scaling CV script, X_test is actually the fold's val set.)
        X_train_l = X_l
        y_train_l = y_l
        X_val = X_test
        y_val = y_test

    input_shape = X_train_l.shape[1:]

    # Decide: supervised-only baseline vs SSL
    run_baseline = (
        X_unlabeled_pool is None
        or ssl_iters <= 0
        or confidence_thresh is None
    )

    if run_baseline:
        # ===========================
        # Supervised-only BASELINE
        # ===========================
        print("[run_pseudo_label_ssl_fold] Running supervised-only baseline (no SSL).")

        # One-hot labels
        y_train_oh = tf.keras.utils.to_categorical(
            y_train_l, num_classes=n_classes
        )
        y_val_oh = tf.keras.utils.to_categorical(
            y_val, num_classes=n_classes
        )

        model = build_model_fn(input_shape, n_classes=n_classes)

        hist = model.fit(
            X_train_l,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs_per_iter,
            batch_size=batch_size,
            verbose=0,
        )

        # Final evaluation on TEST set
        probs = model.predict(X_test, verbose=0)
        y_pred = probs.argmax(axis=1)
        OA_test = (y_pred == y_test).mean()

        # History record to keep interface consistent
        val_probs = model.predict(X_val, verbose=0)
        y_val_pred = val_probs.argmax(axis=1)
        val_acc = (y_val_pred == y_val).mean()

        history = [{
            "iteration": 0,
            "train_size": int(X_train_l.shape[0]),
            "unlabeled_size": 0,
            "n_new_pseudo": 0,
            "val_acc": float(val_acc),
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        }]

        return {
            "OA_test": float(OA_test),
            "history": history,
        }

    # ===========================
    # SSL MODE: call iterative_pseudo_labeling
    # ===========================
    print("[run_pseudo_label_ssl_fold] Running SSL with pseudo-labeling.")

    model, history = iterative_pseudo_labeling(
        model_fn=build_model_fn,
        X_l=X_train_l,
        y_l=y_train_l,
        X_u=X_unlabeled_pool,
        X_val=X_val,
        y_val=y_val,
        n_classes=n_classes,
        n_iters=ssl_iters,
        confidence_threshold=confidence_thresh,
        batch_size=batch_size,
        epochs_per_iter=epochs_per_iter,
        verbose=0,
    )

    # Final evaluation on TEST set
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    OA_test = (y_pred == y_test).mean()

    return {
        "OA_test": float(OA_test),
        "history": history,
    }
