#!/bin/bash
#SBATCH --job-name=summarize_lite_fullbands
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Hyper3DNet-Lite Full-Bands Summary Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

# Load environment
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Ensure Python finds the local package
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Run summary script
python3 experiments/hyper3dnetlite/summarize_lite_fullbands.py

echo "=== Summary Job Completed Successfully ==="
