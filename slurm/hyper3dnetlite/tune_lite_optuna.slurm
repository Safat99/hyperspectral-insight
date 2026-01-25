#!/bin/bash
#SBATCH --job-name=tune_lite_optuna
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

#SBATCH --array=0-19
#SBATCH --output=slurm_logs/bayesian_opt/%x-%A_%a.out
#SBATCH --error=slurm_logs/bayesian_opt/%x-%A_%a.err

# ===================== USER SETTINGS =====================
DATASET="salinas"
TOTAL_TRIALS=40
TRIALS_PER_JOB=2           # 20 × 2 = 40 trials
EPOCHS=30
CV_SPLITS=2
MAX_SAMPLES=2000
N_STARTUP_TRIALS=5
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
echo "Total trials goal:  $TOTAL_TRIALS"
echo "Trials per job:     $TRIALS_PER_JOB"
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
