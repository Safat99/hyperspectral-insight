#!/bin/bash
#SBATCH --job-name=shallow_all
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Running ALL Shallow CNN Baselines ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

# TensorFlow 2.20 compatible CUDA
module load CUDA/12.2.0

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python: $(which python)"
echo "Starting multi-dataset shallow CNN runs..."

python3 experiments/baseline/run_all_datasets.py

echo "=== ALL DATASETS COMPLETED ==="
