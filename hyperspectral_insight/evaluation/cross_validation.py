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
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
    y_val_oh = enc.transform(y_val.reshape(-1, 1))

    # Build model
    model = model_fn(input_shape=X_train.shape[1:], n_classes=n_classes)

    # Train model
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    
    # Convert to plain dict for saving
    history_dict = {
        k : list(v) for k, v in history.history.items()
    }

    # Predict fold accuracy
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    fold_metrics = compute_metrics(y_val, y_pred)

    return model, history_dict, fold_metrics

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
    max_samples_per_class=None,
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
            - training history
    """

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold_results = []
    histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n[CV] Fold {fold_idx + 1}/{n_splits}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        X_train, y_train = cap_samples_per_class(
            X_train,
            y_train,
            max_samples_per_class
            )

        _, history_dict, metrics = train_one_fold(
            model_fn,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        
        
        fold_results.append(metrics)
        histories.append(history_dict)
        
        print(" Fold Metrics:", metrics)

    # Aggregate results
    keys = fold_results[0].keys()
    mean_metrics = {k: float(np.mean([d[k] for d in fold_results])) for k in keys}
    std_metrics = {k: float(np.std([d[k] for d in fold_results])) for k in keys}

    return {
        "fold_metrics": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "histories" : histories,
    }


def cap_samples_per_class(X, y, max_samples_per_class):
        if max_samples_per_class is None:
            return X, y

        X_new, y_new = [], []

        for cls in np.unique(y):
            idx = np.where(y == cls)[0]

            if len(idx) > max_samples_per_class:
                idx = np.random.choice(
                    idx,
                    max_samples_per_class,
                    replace=False
                )

            X_new.append(X[idx])
            y_new.append(y[idx])

        return np.concatenate(X_new), np.concatenate(y_new)