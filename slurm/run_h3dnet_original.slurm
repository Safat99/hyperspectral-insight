#!/bin/bash
#SBATCH --job-name=h3dnet_full
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== Hyper3DNet FULL-BAND CV Started ====="

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/hyper3dnet/run_hyper3dnet_original_cv.py \
    --dataset indian_pines \
    --patch 25 \
    --splits 10 \
    --epochs 50 \
    --batch_size 32

echo "===== Hyper3DNet FULL Completed ====="
