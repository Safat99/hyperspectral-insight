#!/bin/bash
#SBATCH --job-name=conv3d_full
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== Hyper3DNet FULL-BAND CV Started ====="
echo "Dataset: $1"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Submit Dir: $SLURM_SUBMIT_DIR"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/conv3d_full/run_conv3d_full.py \
    --dataset "$1" \
    --patch 25 \
    --splits 10 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4

echo "===== Hyper3DNet FULL Completed ====="
