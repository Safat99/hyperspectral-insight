#!/bin/bash
#SBATCH --job-name=tune_shallow_cnn
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Starting Shallow CNN Tuning ==="

cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 experiments/baseline/tune_shallow_cnn.py --dataset indian_pines

echo "=== Tuning Completed ==="
