#!/bin/bash
#SBATCH --job-name=ssl_supervised
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ssl/supervised_%A_%a.out
#SBATCH --error=logs/ssl/supervised_%A_%a.err
#SBATCH --array=0-2

DATASETS=("indian_pines" "salinas" "pavia")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running FULL supervised baseline on $DATASET"

source ~/.bashrc
conda activate hyperspectral-insight

mkdir -p logs/ssl/

python experiments/ssl/run_supervised_full_labeled_baseline.py \
    --dataset $DATASET \
    --num_bands 20 \
    --patch 25 \
    --splits 10 \
    --epochs 50 \
    --batch_size 16

echo "Supervised baseline complete: $DATASET"
