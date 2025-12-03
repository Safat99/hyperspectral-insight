#!/bin/bash
#SBATCH --job-name=h3dnet_pca
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== Hyper3DNet PCA CV Started ====="
echo "Dataset: $1"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/hyper3dnet/run_hyper3dnet_pca_cv.py \
    --dataset "$1" \
    --patch 25 \
    --var_thresh 0.99 \
    --max_components 30 \
    --splits 10 \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4

echo "===== Hyper3DNet PCA Completed ====="
