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
    patch_stride_map,
    tuning_cv=2,
    tuning_epochs=30,
    max_samples=2000,
):
    """
    Returns an Optuna objective function.
    Architecture is injected via model_builder.
    """

    def objective(trial):

        optimizer = trial.suggest_categorical("optimizer", optimizer_space)
        
        patch_key = trial.suggest_categorical(
            "patch_stride", list(patch_stride_map.keys())
        )
        patch_size, stride = patch_stride_map[patch_key]
        
        # ---------------- conditional batch size --------------------
        
        # if patch_size >= 50:
        #     batch_size = trial.suggest_categorical("batch_size", [8, 16])
        # elif patch_size >= 33:
        #     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        # else:
        #     batch_size = trial.suggest_categorical(
        #         "batch_size", [8, 16, 32, 64, 128]
        #     )
        
        # -------- Static batch size space (Optuna-safe) --------
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128]
        )

        # -------- Prune illegal combinations --------
        if patch_size >= 50 and batch_size > 16:
            raise optuna.TrialPruned("Batch too large for patch >= 50")

        if patch_size >= 33 and batch_size > 32:
            raise optuna.TrialPruned("Batch too large for patch >= 33")
        
        
        # ------------------- Optimizer-specific hyperparameters -----------------
    
        if optimizer == "adam":
            opt_params = {
                "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            }
        else:  # adadelta
            opt_params = {
                "rho": trial.suggest_float("rho", 0.85, 0.99),
                "epsilon": trial.suggest_float(
                    "epsilon", 1e-7, 1e-3, log=True
                ),
            }
        # ----------------------------------------------------------------------
            
        X, y = extract_patches(
            cube,
            gt,
            win=patch_size,
            stride=stride,
            max_samples_per_class=None
        )
        
        # ---------------- Memory safety prune ----------------
        # Large patches on large datasets can exceed host RAM
        if patch_size >= 50 and X.shape[0] > 50_000:
            raise optuna.TrialPruned(
                f"Memory-unsafe config: patch={patch_size}, n_patches={X.shape[0]}"
            )

        # ----- Model factory -----
        def model_fn(input_shape, n_classes):
            return model_builder(
                input_shape,
                n_classes,
                optimizer=optimizer,
                opt_params=opt_params,
            )

        # ------------------- CV sanity check --------------------------
        class_counts = Counter(y)
        min_class = min(class_counts.values())
        effective_splits = min(tuning_cv, min_class)

        if effective_splits < 2:
            raise optuna.TrialPruned("Too few samples for CV")
        # -----------------------------------------------------------------
        
        # ----- CV evaluation (tuning mode)-----
        results = kfold_cross_validation(
            model_fn=model_fn,
            X=X,
            y=y,
            n_splits=effective_splits,
            epochs=tuning_epochs,
            batch_size=batch_size,
            max_samples_per_class=max_samples,
            verbose=1,
            use_early_stopping=True,
        )
        
        # ---------------- Store secondary metrics ----------------
        mean_metrics = results["mean_metrics"]
        trial.set_user_attr("oa", mean_metrics.get("oa"))
        trial.set_user_attr("kappa", mean_metrics.get("kappa"))
        trial.set_user_attr(
            "trained_epochs",
            results.get("mean_trained_epochs")
        )
        
        # ---------------- Optimization target ----------------
        return results["mean_metrics"]["f1"]

    return objective
