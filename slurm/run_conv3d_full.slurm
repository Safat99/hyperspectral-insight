#!/bin/bash
#SBATCH --job-name=conv3d_full
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

#SBATCH --array=0-4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Conv3d FULL-BAND CV Started ====="

# DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running dataset: $DATASET"
echo "Array task: $SLURM_ARRAY_TASK_ID"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs


python3 experiments/conv3d_full/run_conv3d_full.py \
    --dataset "$DATASET" \
    --patch 25 \
    --splits 5 \
    --epochs 50 \
    --batch_size 128 \
    --max_samples 2000 \
    --learning_rate 5e-4


