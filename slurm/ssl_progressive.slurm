#!/bin/bash
#SBATCH --job-name=ssl_progressive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl/progressive_%A_%a.out
#SBATCH --error=logs/ssl/progressive_%A_%a.err
#SBATCH --array=0-2

DATASETS=("indian_pines" "salinas" "pavia")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running SSL Progressive Unlabeled for dataset: $DATASET"

source ~/.bashrc
conda activate hyperspectral-insight

mkdir -p logs/ssl/

python experiments/ssl/run_ssl_progressive_unlabeled.py \
    --dataset $DATASET \
    --num_bands 20 \
    --patch 25 \
    --ssl_iters 5 \
    --conf 0.90 \
    --epochs 20 \
    --batch 16

echo "Progressive SSL completed for $DATASET"
