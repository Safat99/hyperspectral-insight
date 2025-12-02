#!/bin/bash
#SBATCH --job-name=shallow_baseline
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Starting Shallow CNN Baseline ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Submit Dir: $SLURM_SUBMIT_DIR"

# Go to working directory
cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

# Load CUDA 12.2 for tensorflow 2.20
module load CUDA/12.2.0

# Load Anaconda and activate your env
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Make sure Python can import your local package
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python version: $(python3 --version)"
echo "Running baseline for dataset = indian_pines"

python3 experiments/baseline/run_shallow_cnn.py --dataset indian_pines

echo "=== Job Completed ==="

