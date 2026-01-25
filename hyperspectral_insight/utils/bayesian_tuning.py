import optuna
from hyperspectral_insight.data.patches import extract_patches
from hyperspectral_insight.evaluation.cross_validation import kfold_cross_validation
from collections import Counter

def make_objective(
    *,
    cube,
    gt,
    model_builder,
    optimizer_space,
    batch_space,
    patch_stride_map,
    tuning_cv=5,
    tuning_epochs=10,
    max_samples=1000,
    safety_fn=None,
):
    """
    Returns an Optuna objective function.
    Architecture is injected via model_builder.
    """

    def objective(trial):

        # ----- Search space -----
        optimizer = trial.suggest_categorical("optimizer", optimizer_space)
        batch_size = trial.suggest_categorical("batch_size", batch_space)
        patch_key = trial.suggest_categorical(
            "patch_stride", list(patch_stride_map.keys()
                                 )
        )
        patch_size, stride = patch_stride_map[patch_key]
        
        # ----- Safety / pruning -----
        if safety_fn is not None:
            if not safety_fn(patch_size, stride, batch_size, optimizer):
                raise optuna.TrialPruned("Unsafe configuration")

        
        X, y = extract_patches(
            cube,
            gt,
            win=patch_size,
            stride=stride,
            max_samples_per_class=max_samples
        )

        # ----- Model factory -----
        def model_fn(input_shape, n_classes):
            return model_builder(
                input_shape,
                n_classes,
                optimizer=optimizer
            )

        class_counts = Counter(y)
        min_class = min(class_counts.values())
        effective_splits = min(tuning_cv, min_class)

        epochs = tuning_epochs
        if optimizer == "adadelta":
            effective_splits = min(2, effective_splits)
            epochs = max(epochs, 20)    
        
        if effective_splits < 2:
            raise optuna.TrialPruned("Too few samples for CV")
        
        # ----- CV evaluation -----
        results = kfold_cross_validation(
            model_fn=model_fn,
            X=X,
            y=y,
            n_splits=effective_splits,
            epochs=epochs,
            batch_size=batch_size,
            max_samples_per_class=max_samples,
            verbose=1,
        )

        return results["mean_metrics"]["f1"]

    return objective
