#!/bin/bash
#SBATCH --job-name=tune_shallow_cnn
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/new/%x-%j.out
#SBATCH --error=slurm_logs/new/%x-%j.err

echo "=== Starting Shallow CNN Tuning ==="

cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 experiments/baseline/tune_shallow_cnn.py --dataset pavia_centre

echo "=== Tuning Completed ==="
