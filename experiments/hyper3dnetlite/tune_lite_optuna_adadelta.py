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


def tune_lite_optuna_adadelta(
    dataset_name: str,
    trials_per_job: int = 1,
    out_dir: str = "results/hyper3dnetlite/new/adadelta_optuna",
    tuning_cv: int = 2,
    tuning_epochs: int = 15,
    max_samples: int = 1000,
):

    print(f"\n=== Adadelta Optuna (FAST & FAIR) | Dataset={dataset_name} ===")
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------- Trial log CSV ----------
    csv_path = os.path.join(out_dir, f"{dataset_name}_adadelta_trials.csv")

    # ---------- Dataset ----------
    cube, gt = load_dataset(dataset_name)
    cube = minmax_normalize(cube)

    # ---------- Model builder ----------
    def model_builder(input_shape, n_classes, optimizer):
        return build_hyper3dnet_lite(
            input_shape,
            n_classes,
            optimizer_name="adadelta"
        )

    # ---------- Safety rule (OOM only) ----------
    def safety_rule(patch, stride, batch, optimizer):
        # Adadelta-only, conservative
        if patch == 50 and batch >= 64:
            return False
        return True

    # ---------- FIXED patch & batch (paper-faithful) ----------
    PATCH_STRIDE_MAP = {
        "p5_s1": (5, 1),
        "p25_s1": (25, 1),
        "p50_s1": (50, 1),
    }

    batch_space = [16, 64, 128]
    optimizer_space = ["adadelta"]

    # ---------- Objective ----------
    objective = make_objective(
        cube=cube,
        gt=gt,
        model_builder=model_builder,
        optimizer_space=optimizer_space,
        batch_space=batch_space,
        patch_stride_map=PATCH_STRIDE_MAP,
        tuning_cv=tuning_cv,           # 2-fold
        tuning_epochs=tuning_epochs,   # 15 epochs
        max_samples=max_samples,
        safety_fn=safety_rule,
    )

    # ---------- Storage ----------
    storage_path = os.path.join(out_dir, f"optuna_{dataset_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"h3dnetlite_{dataset_name}_adadelta"

    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    # sampler = optuna.samplers.RandomSampler(seed=seed)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=10,   # first 10 trials = random
        multivariate=True,
    )
    
    # ---------- Study (IMPORTANT) ----------
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )
    
    def on_trial_complete(study, trial):
        # Console log (goes to SLURM .out)
        print(
            f"[TRIAL {trial.number}] "
            f"state={trial.state.name} "
            f"value={trial.value} "
            f"params={trial.params}"
        )

        # Append to CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "timestamp",
                    "trial_number",
                    "state",
                    "value",
                    "optimizer",
                    "batch_size",
                    "patch_stride",
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                trial.number,
                trial.state.name,
                trial.value,
                trial.params.get("optimizer"),
                trial.params.get("batch_size"),
                trial.params.get("patch_stride"),
            ])

    # ---------- Run ----------
    # ---------- Run ----------
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts: 
        study.optimize(
            objective,
            n_trials=trials_per_job,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
            callbacks=[on_trial_complete],
        )

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if completed_trials:
            best_trial = study.best_trial
            best_f1 = best_trial.value
            best_params = best_trial.params
            break
        else:
            best_f1 = None
            best_params = None
            print(" No completed trials yet (all trials pruned).")
            attempt +=1

    result = {
        "dataset": dataset_name,
        "optimizer": "adadelta",
        "best_f1": best_f1,
        "best_params": best_params,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "tuning_epochs": tuning_epochs,
        "tuning_cv": tuning_cv,
        "max_samples": max_samples,
    }

    out_path = os.path.join(out_dir, f"{dataset_name}_adadelta_best.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print("\n=== DONE (Adadelta) ===")
    print(" Trials:", len(study.trials))

    if completed_trials:
        print(" Best F1:", best_f1)
    else:
        print(" Best F1: N/A (no completed trials yet)")

    print(" DB:", storage_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials_per_job", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="results/hyper3dnetlite/new/adadelta_optuna")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=1000)

    args = parser.parse_args()

    tune_lite_optuna_adadelta(
        dataset_name=args.dataset,
        trials_per_job=args.trials_per_job,
        out_dir=args.out_dir,
        tuning_cv=args.splits,
        tuning_epochs=args.epochs,
        max_samples=args.max_samples,
    )
