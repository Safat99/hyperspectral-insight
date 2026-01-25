import argparse
import json
import os
import optuna

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.utils.bayesian_tuning import make_objective


def tune_lite_optuna(
    dataset_name: str,
    # n_trials: int = 30,
    trials_per_job: int = 2,
    out_dir: str = "results/hyper3dnetlite/new/optuna",
    tuning_cv: int = 5,
    tuning_epochs: int = 10,
    max_samples: int = 1000,
):
    """
    Bayesian hyperparameter tuning for Hyper3DNet-Lite
    using Optuna + stratified CV.
    """

    print(f"\n=== Optuna tuning: Hyper3DNet-Lite | Dataset={dataset_name} ===")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Dataset loader ----------
    
    cube, gt = load_dataset(dataset_name)
    cube = minmax_normalize(cube)

    
    # ---------- Model builder ----------
    def model_builder(input_shape, n_classes, optimizer):
        if optimizer == "adam":
            return build_hyper3dnet_lite(
                input_shape,
                n_classes,
                optimizer_name="adam",
                lr=1e-3
            )
        elif optimizer == "adadelta":
            return build_hyper3dnet_lite(
                input_shape,
                n_classes,
                optimizer_name="adadelta"
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # ---------- Safety rule ----------
    def safety_rule(patch, stride, batch, optimizer):
        # Avoid known OOM configs
        # Known GPU-unsafe configs
        if patch == 50 and batch >= 64:
            return False
        return True

    PATCH_STRIDE_MAP = {
        "p5_s1": (5, 1),
        "p25_s1": (25, 1),
        "p50_s25": (50, 25),
    }
    
    # ---------- Build Optuna objective ----------
    objective = make_objective(
        cube=cube,
        gt=gt,
        model_builder=model_builder,
        optimizer_space=["adam", "adadelta"],
        batch_space=[16, 64, 128],
        patch_stride_map=PATCH_STRIDE_MAP,
        tuning_cv=tuning_cv,
        tuning_epochs=tuning_epochs,
        max_samples=max_samples,
        safety_fn=safety_rule,
    )
    
    # ---------- Shared Optuna storage ----------
    storage_path = os.path.join(out_dir, f"optuna_{dataset_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"h3dnetlite_{dataset_name}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    

    # ---------- Run study ----------
    # study = optuna.create_study(
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=0)
    # )

    study.optimize(
        objective,
        n_trials=trials_per_job,
        n_jobs=1,   # SLURM-safe
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # ---------- Save results ----------
    result = {
        "dataset": dataset_name,
        "study_name": study_name,
        "best_params": study.best_params,
        "best_f1": study.best_value,
        # "n_trials": n_trials,
        "total_trials": len(study.trials),
        "tuning_cv": tuning_cv,
        "tuning_epochs": tuning_epochs,
        "max_samples": max_samples,
    }

    out_path = os.path.join(
        out_dir, f"{dataset_name}_optuna_best.json"
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print("\n=== Optuna tuning progress saved ===")
    print(" Dataset:", dataset_name)
    print(" Trials so far:", len(study.trials))
    print("Best params:", study.best_params)
    print("Best F1:", study.best_value)
    print(" DB:", storage_path)
    # print("Saved to:", out_path)

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials_per_job", type=int, default=2)
    # parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="results/hyper3dnetlite/new/optuna")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=1000)

    args = parser.parse_args()

    tune_lite_optuna(
        dataset_name=args.dataset,
        trials_per_job=args.trials_per_job,
        out_dir=args.out_dir,
        tuning_cv=args.splits,
        tuning_epochs=args.epochs,
        max_samples=args.max_samples,
    )
