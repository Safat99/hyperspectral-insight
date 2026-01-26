#!/bin/bash
#SBATCH --job-name=tune_lite_optuna
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=23:00:00

#SBATCH --array=0-7
#SBATCH --output=slurm_logs/bayesian_opt/%x-%A_%a.out
#SBATCH --error=slurm_logs/bayesian_opt/%x-%A_%a.err

# ===================== USER SETTINGS =====================
DATASET="salinas"

TRIALS_PER_JOB=3          # safe against mid-trial timeout
EPOCHS=30                 # full training
CV_SPLITS=2
MAX_SAMPLES=2000
N_STARTUP_TRIALS=10
# =========================================================

# ---------------- Environment ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"

# ---------------- Logging ----------------
mkdir -p slurm_logs/bayesian_opt

echo "========================================"
echo "Dataset:            $DATASET"
echo "Trials per job:     $TRIALS_PER_JOB"
echo "Epochs:             $EPOCHS"
echo "CV splits:          $CV_SPLITS"
echo "Array task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:               $(hostname)"
echo "========================================"

# ---------------- Run Optuna tuning ----------------
python experiments/hyper3dnetlite/tune_lite_optuna.py \
    --dataset "$DATASET" \
    --trials_per_job $TRIALS_PER_JOB \
    --epochs $EPOCHS \
    --splits $CV_SPLITS \
    --max_samples $MAX_SAMPLES \
    --n_startup_trials $N_STARTUP_TRIALS
