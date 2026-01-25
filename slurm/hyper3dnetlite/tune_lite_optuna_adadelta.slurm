#!/bin/bash
#SBATCH --job-name=adadelta_optuna
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

#SBATCH --output=slurm_logs/adadelta_optuna/%x-%A_%a.out
#SBATCH --error=slurm_logs/adadelta_optuna/%x-%A_%a.err

# ===== Job array: datasets × trials =====
# 6 trials total per dataset → array 0–5
#SBATCH --array=0-9

# ---------- Dataset selection ----------
DATASET="indian_pines"     # change as needed

# ---------- Environment ----------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# ---------- Paths ----------
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/adadelta_optuna

echo "=============================="
echo "Dataset: $DATASET"
echo "SLURM job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "=============================="

# ---------- Run ONE Optuna trial ----------
python experiments/hyper3dnet_lite/tune_lite_optuna_adadelta.py \
    --dataset "$DATASET" \
    --trials_per_job 1 \
    --epochs 30 \
    --splits 2 \
    --max_samples 1000
