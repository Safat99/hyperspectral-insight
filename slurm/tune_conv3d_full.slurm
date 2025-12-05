#!/bin/bash
#SBATCH --job-name=tune_conv3d_full
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

#SBATCH --array=0-4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== Conv3d FULL-BAND CV Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Submit Dir: $SLURM_SUBMIT_DIR"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/conv3d_full/tune_conv3d_full.py \
    --dataset "$DATASET" \
    --splits 3 \
    --epochs 20 \
    --max_samples 2000

echo "===== Conv3d FULL Completed ====="
