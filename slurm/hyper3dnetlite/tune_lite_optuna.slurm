#!/bin/bash
#SBATCH --job-name=tune_lite_optuna
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

#SBATCH --output=slurm_logs/bayesian_opt/%x-%A_%a.out
#SBATCH --error=slurm_logs/bayesian_opt/%x-%A_%a.err
#SBATCH --array=0-4

DATASETS=("indian_pines" "salinas" "pavia_centre" "pavia_university" "ksc")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

echo "Dataset: $DATASET"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python experiments/hyper3dnetlite/tune_lite_optuna.py \
    --dataset "$DATASET" \
    --trials 30
