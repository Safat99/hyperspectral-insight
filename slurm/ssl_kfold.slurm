#!/bin/bash
#SBATCH --job-name=ssl_kfold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl/kfold_%A_%a.out
#SBATCH --error=logs/ssl/kfold_%A_%a.err
#SBATCH --array=0-2   # Indian Pines, Salinas, Pavia

# --------------------------
# Dataset list for array
# --------------------------
DATASETS=("indian_pines" "salinas" "pavia")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running SSL K-Fold for dataset: $DATASET"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Activate Conda env
source ~/.bashrc
conda activate hyperspectral-insight

# Create log directory if missing
mkdir -p logs/ssl/

# Run experiment
python experiments/ssl/run_ssl_lite_ibra_gss_kfold.py \
    --dataset $DATASET \
    --num_bands 20 \
    --patch 25 \
    --splits 10 \
    --ssl_iters 5 \
    --epochs 20 \
    --batch_size 16 \
    --conf 0.90

echo "Finished SSL K-fold for $DATASET"
