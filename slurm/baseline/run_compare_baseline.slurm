#!/bin/bash
#SBATCH --job-name=analyze_baseline
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Compare Baselines: Summary Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Python import path fix (important!)
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

python3 experiments/baseline/compare_baselines.py

echo "=== Summary Completed ==="
