import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from hyperspectral_insight.evaluation.metrics import compute_metrics


def train_one_fold(
    model_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=16,
    verbose=0,
):
    """
    Train a single fold of a supervised model.

    Args:
        model_fn: function returning a compiled Keras model.
        X_train: numpy array of training patches.
        y_train: integer labels.
        X_val: validation patches.
        y_val: integer labels.
        epochs: training epochs.
        batch_size: mini-batch size.
        verbose: Keras verbosity.

    Returns:
        model: trained model
        history: Keras training history object
        metrics: dictionary containing OA, Precision, Recall, F1, Kappa
    """

    n_classes = int(y_train.max() + 1)
    """y_train_oh = OneHotEncoder(sparse_output=False).fit_transform(
        y_train.reshape(-1, 1)
    )
    y_val_oh = OneHotEncoder(sparse_output=False).fit_transform(
        y_val.reshape(-1, 1)
    )"""

    model = model_fn(input_shape=X_train.shape[1:], n_classes=n_classes)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    fold_metrics = compute_metrics(y_val, y_pred)

    return model, history, fold_metrics

def kfold_cross_validation(
    model_fn,
    X,
    y,
    n_splits=5,
    epochs=30,
    batch_size=16,
    shuffle=True,
    random_state=0,
    verbose=0,
):
    """
    Perform K-fold cross-validation on a patch dataset.

    Args:
        model_fn: model-building function
        X, y: data and labels
        n_splits: number of folds
        epochs, batch_size: Keras training parameters
        shuffle: enable KFold shuffle
        random_state: RNG seed for Kfold
        verbose: training verbosity

    Returns:
        results: dict with:
            - fold_metrics: list of metric dicts
            - mean_metrics
            - std_metrics
    """

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n[CV] Fold {fold_idx + 1}/{n_splits}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        _, _, metrics = train_one_fold(
            model_fn,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        print(" Fold Metrics:", metrics)
        fold_results.append(metrics)

    # Aggregate results
    keys = fold_results[0].keys()
    mean_metrics = {k: float(np.mean([d[k] for d in fold_results])) for k in keys}
    std_metrics = {k: float(np.std([d[k] for d in fold_results])) for k in keys}

    return {
        "fold_metrics": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }
