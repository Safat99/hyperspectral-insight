import argparse
import json
import os
import optuna
import csv
from datetime import datetime

from hyperspectral_insight.data.loaders import load_dataset
from hyperspectral_insight.data.normalization import minmax_normalize
from hyperspectral_insight.models.hyper3dnet_lite import build_hyper3dnet_lite
from hyperspectral_insight.utils.bayesian_tuning import make_objective


def tune_lite_optuna(
    dataset_name: str,
    trials_per_job: int = 2,
    out_dir: str = "results/hyper3dnetlite/new/optuna",
    tuning_cv: int = 2,
    tuning_epochs: int = 30,
    max_samples: int = 2000,
    n_startup_trials: int = 5,
    write_task_csv: bool = True,
):

    print(f"\n=== Optuna tuning | Dataset={dataset_name} ===")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{dataset_name}_trials.csv")

    # ---------- Dataset ----------
    cube, gt = load_dataset(dataset_name)
    cube = minmax_normalize(cube)

    # ---------- Model builder ----------
    def model_builder(input_shape, n_classes, optimizer, opt_params):
        return build_hyper3dnet_lite(
            input_shape=input_shape,
            n_classes=n_classes,
            optimizer_name=optimizer,
            **opt_params,
        )

    # ---------- Safety rule ----------
    def safety_rule(patch, stride, batch, optimizer):
        if patch == 50 and batch >= 64:
            return False
        return True

    PATCH_STRIDE_MAP = {
        "p5_s1": (5, 1),
        "p9_s1": (9, 1),
        "p13_s1": (13, 1),
        "p17_s1": (17, 1),
        "p25_s1": (25, 1),
        "p33_s2": (33, 2),
        "p50_s2": (50, 2)
    }

    # ---------- Objective ----------
    objective = make_objective(
        cube=cube,
        gt=gt,
        model_builder=model_builder,
        optimizer_space=["adam", "adadelta"],
        patch_stride_map=PATCH_STRIDE_MAP,
        tuning_cv=tuning_cv,
        tuning_epochs=tuning_epochs,
        max_samples=max_samples,
    )

    # ---------- Storage (shared across array jobs)------------------
    storage_path = os.path.join(out_dir, f"optuna_{dataset_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"h3dnetlite_{dataset_name}"

    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        # consider_pruned_trials=False,
    )

    # ---------- Logging ----------
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    csv_path = os.path.join(out_dir, f"{dataset_name}_trials_task{task_id}.csv")
    
    def on_trial_complete(study, trial):
        print(
            f"[TRIAL {trial.number}] "
            f"state={trial.state.name} "
            f"value={trial.value} "
            f"params={trial.params} "
            f"user_attrs={trial.user_attrs}"
        )

        if not write_task_csv:
            return
        
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    "timestamp",
                    "trial",
                    "state",
                    "value_f1",
                    "oa",
                    "kappa",
                    "trained_epochs",
                    "optimizer",
                    "batch_size",
                    "patch_stride",
                    "lr",
                    "rho",
                    "epsilon",
                ])
                
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                trial.number,
                trial.state.name,
                trial.value,
                trial.user_attrs.get("oa"),
                trial.user_attrs.get("kappa"),
                trial.user_attrs.get("trained_epochs"),
                trial.params.get("optimizer"),
                trial.params.get("batch_size"),
                trial.params.get("patch_stride"),
                trial.params.get("lr"),
                trial.params.get("rho"),
                trial.params.get("epsilon"),
            ])

    # ---------- Run ----------
    study.optimize(
        objective,
        n_trials=trials_per_job,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[on_trial_complete],
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    best_trial = study.best_trial if completed else None
    
    result = {
        "dataset": dataset_name,
        "best_f1": best_trial.value if best_trial else None,
        "best_params": best_trial.params if best_trial else None,
        "best_oa": best_trial.user_attrs.get("oa") if best_trial else None,
        "best_kappa": best_trial.user_attrs.get("kappa") if best_trial else None,
        "best_trained_epochs": best_trial.user_attrs.get("trained_epochs") if best_trial else None,
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "tuning_epochs_cap": tuning_epochs,
        "tuning_cv": tuning_cv,
        "max_samples": max_samples,
        "storage_db": storage_path,
        "study_name": study_name,
    }

    out_path = os.path.join(out_dir, f"{dataset_name}_optuna_best.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print("\n=== DONE ===")
    print(" Trials:", len(study.trials))
    print(" Completed:", len(completed))
    print(" Best F1:", result["best_f1"])
    print(" Best OA:", result["best_oa"])
    print(" DB:", storage_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials_per_job", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results/hyper3dnetlite/new/optuna")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--n_startup_trials", type=int, default=5)
    parser.add_argument("--no_task_csv", action="store_true")

    args = parser.parse_args()

    tune_lite_optuna(
        dataset_name=args.dataset,
        trials_per_job=args.trials_per_job,
        out_dir=args.out_dir,
        tuning_cv=args.splits,
        tuning_epochs=args.epochs,
        max_samples=args.max_samples,
        n_startup_trials=args.n_startup_trials,
        write_task_csv=(not args.no_task_csv),
    )