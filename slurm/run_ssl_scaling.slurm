#!/bin/bash
#SBATCH --job-name=ssl_scaling
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== SSL Scaling Experiment Started ====="
echo "Dataset: $1"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 run_ssl_scaling_experiment.py \
    --dataset "$1"

echo "===== SSL Scaling Experiment Completed ====="
