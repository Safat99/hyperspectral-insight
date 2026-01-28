# hyperspectral_insight/ssl/pseudo_label.py

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Optional

from hyperspectral_insight.ssl.ssl_utils import merge_labeled_and_pseudo
from hyperspectral_insight.evaluation.metrics import compute_metrics  # <-- NEW


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

    probs_list = []
    for i in range(0, X.shape[0], batch_size):
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
    """
    if X_u is None or X_u.shape[0] == 0:
        # consistent empty outputs
        return (
            np.empty((0,) + (X_u.shape[1:] if X_u is not None else ()), dtype=(X_u.dtype if X_u is not None else np.float32)),
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

    IMPORTANT FAIRNESS FIX:
    - We match training effort to the supervised baseline by using a fixed
      "sample budget" equal to (epochs_per_iter * N_labeled_initial).
    - As the SSL training set grows, we reduce epochs dynamically so total
      samples processed stays comparable.
    """

    if n_classes is None:
        n_classes = int(y_l.max() + 1)

    input_shape = X_l.shape[1:]

    # One-hot encode labels
    y_l_oh = tf.keras.utils.to_categorical(y_l, num_classes=n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    run_ssl = (
        X_u is not None
        and X_u.shape[0] > 0
        and n_iters > 0
        and confidence_threshold is not None
    )

    # Build model once; keep training across iterations
    model = model_fn(input_shape, n_classes=n_classes)

    X_train = X_l.copy()
    y_train_oh = y_l_oh.copy()

    if X_u is not None and X_u.shape[0] > 0:
        X_unlabeled = X_u.copy()
    else:
        X_unlabeled = np.empty((0,) + X_l.shape[1:], dtype=X_l.dtype)

    history: List[Dict] = []

    # -----------------------------
    # Supervised-only fallback
    # -----------------------------
    if not run_ssl:
        print("[iterative_pseudo_labeling] No valid unlabeled pool or SSL config → supervised-only training.")

        hist = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs_per_iter,
            batch_size=batch_size,
            verbose=verbose,
        )

        val_probs = model.predict(X_val, verbose=0)
        y_val_pred = val_probs.argmax(axis=1)
        val_acc = float((y_val_pred == y_val).mean())

        history.append({
            "iteration": 0,
            "mode": "supervised_only",
            "train_size": int(X_train.shape[0]),
            "unlabeled_size": int(X_unlabeled.shape[0]),
            "n_new_pseudo": 0,
            "epochs_this_iter": int(epochs_per_iter),
            "budget_samples_total": int(epochs_per_iter * X_train.shape[0]),
            "budget_samples_used": int(epochs_per_iter * X_train.shape[0]),
            "val_acc": val_acc,
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        })
        return model, history

    # ==========================================================
    # FAIRNESS: Budget in "samples processed"
    # Baseline effort per fold ≈ epochs_per_iter * N_initial_labeled
    # ==========================================================
    baseline_budget_samples = int(epochs_per_iter * X_train.shape[0])
    used_samples = 0

    print(f"[SSL] Baseline sample budget: {baseline_budget_samples} (={epochs_per_iter} epochs × {X_train.shape[0]} samples)")

    # -----------------------------
    # SSL pseudo-label iterations
    # -----------------------------
    for it in range(n_iters):
        print(f"\n[SSL] Iteration {it + 1}/{n_iters}")
        print(f"  Train size: {X_train.shape[0]}  |  Unlabeled size: {X_unlabeled.shape[0]}  |  Used budget: {used_samples}/{baseline_budget_samples}")

        remaining = baseline_budget_samples - used_samples
        if remaining <= 0:
            print("  Training budget exhausted; stopping SSL training.")
            break

        # Dynamic epochs: keep total processed samples comparable
        train_size_now = int(X_train.shape[0])
        epochs_this_iter = max(1, remaining // train_size_now)

        # Safety: do not exceed the original epochs_per_iter in one shot
        epochs_this_iter = int(min(epochs_this_iter, epochs_per_iter))

        print(f"  Epochs this iter (dynamic): {epochs_this_iter}")

        hist = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs_this_iter,
            batch_size=batch_size,
            verbose=verbose,
        )

        used_samples += int(epochs_this_iter * train_size_now)

        # Evaluate on validation
        val_probs = model.predict(X_val, verbose=0)
        y_val_pred = val_probs.argmax(axis=1)
        val_acc = float((y_val_pred == y_val).mean())

        # If unlabeled exhausted, log and stop
        if X_unlabeled.shape[0] == 0:
            print("  Unlabeled pool exhausted; stopping.")
            history.append({
                "iteration": it,
                "mode": "ssl",
                "train_size": int(X_train.shape[0]),
                "unlabeled_size": 0,
                "n_new_pseudo": 0,
                "epochs_this_iter": int(epochs_this_iter),
                "budget_samples_total": int(baseline_budget_samples),
                "budget_samples_used": int(used_samples),
                "val_acc": val_acc,
                "last_epoch_train_loss": float(hist.history["loss"][-1]),
                "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
            })
            break

        # Generate pseudo labels
        X_pseudo, y_pseudo, confidences, selected_idx = generate_pseudo_labels(
            model,
            X_unlabeled,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
        )

        n_new = int(X_pseudo.shape[0])

        if max_pseudo_per_iter is not None and n_new > max_pseudo_per_iter:
            order = np.argsort(-confidences)
            topk = order[:max_pseudo_per_iter]
            X_pseudo = X_pseudo[topk]
            y_pseudo = y_pseudo[topk]
            selected_idx = selected_idx[topk]
            n_new = int(X_pseudo.shape[0])

        print(f"  New pseudo-labels accepted: {n_new}")

        history.append({
            "iteration": it,
            "mode": "ssl",
            "train_size": int(X_train.shape[0]),
            "unlabeled_size": int(X_unlabeled.shape[0]),
            "n_new_pseudo": int(n_new),
            "epochs_this_iter": int(epochs_this_iter),
            "budget_samples_total": int(baseline_budget_samples),
            "budget_samples_used": int(used_samples),
            "val_acc": val_acc,
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        })

        # No new pseudo-labels → stop
        if n_new == 0:
            print("  No new pseudo-labels added; stopping early.")
            break

        # Merge pseudo-labeled samples into training set
        X_train, y_train_oh = merge_labeled_and_pseudo(
            X_train,
            y_train_oh,
            X_pseudo,
            y_pseudo,
            n_classes=n_classes,
        )

        # Remove pseudo-labeled from unlabeled pool
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
    epochs_per_iter: int = 20,   # baseline epoch budget
    batch_size: int = 16,
    confidence_thresh: Optional[float] = 0.9,
    ssl_iters: int = 5,
    val_frac: float = 0.2,
) -> Dict:
    """
    High-level wrapper for a SINGLE FOLD.
    Returns BOTH OA and F1 for consistency with your study.
    """

    N = X_labeled_fold.shape[0]
    n_classes = int(y_labeled_fold.max() + 1)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    X_l = X_labeled_fold[perm]
    y_l = y_labeled_fold[perm]

    # Train/val split inside fold (optional)
    if val_frac > 0.0 and N > 1:
        split = int((1.0 - val_frac) * N)
        split = max(1, min(split, N - 1))
        X_train_l = X_l[:split]
        y_train_l = y_l[:split]
        X_val = X_l[split:]
        y_val = y_l[split:]
    else:
        X_train_l = X_l
        y_train_l = y_l
        X_val = X_test
        y_val = y_test

    input_shape = X_train_l.shape[1:]

    run_baseline = (
        X_unlabeled_pool is None
        or ssl_iters <= 0
        or confidence_thresh is None
    )

    if run_baseline:
        print("[run_pseudo_label_ssl_fold] Running supervised-only baseline (no SSL).")

        y_train_oh = tf.keras.utils.to_categorical(y_train_l, num_classes=n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

        model = build_model_fn(input_shape, n_classes=n_classes)

        hist = model.fit(
            X_train_l,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs_per_iter,
            batch_size=batch_size,
            verbose=0,
        )

        probs = model.predict(X_test, verbose=0)
        y_pred = probs.argmax(axis=1)

        m = compute_metrics(y_test, y_pred)  # expects keys: OA, Precision, Recall, F1, Kappa

        # keep history format consistent
        val_probs = model.predict(X_val, verbose=0)
        y_val_pred = val_probs.argmax(axis=1)
        val_acc = float((y_val_pred == y_val).mean())

        history = [{
            "iteration": 0,
            "mode": "baseline",
            "train_size": int(X_train_l.shape[0]),
            "unlabeled_size": 0,
            "n_new_pseudo": 0,
            "epochs_this_iter": int(epochs_per_iter),
            "val_acc": float(val_acc),
            "last_epoch_train_loss": float(hist.history["loss"][-1]),
            "last_epoch_val_loss": float(hist.history["val_loss"][-1]),
        }]

        return {
            "OA_test": float(m["OA"]),
            "F1_test": float(m["F1"]),
            "Kappa_test": float(m.get("Kappa", 0.0)),
            "history": history,
        }

    # SSL mode
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
        epochs_per_iter=epochs_per_iter,  # baseline epoch budget used to compute sample budget
        verbose=0,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    m = compute_metrics(y_test, y_pred)

    return {
        "OA_test": float(m["OA"]),
        "F1_test": float(m["F1"]),
        "Kappa_test": float(m.get("Kappa", 0.0)),
        "history": history,
    }
