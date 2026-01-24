#!/bin/bash
#SBATCH --job-name=analyze_h3dnetlite
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Analyzing Hyper3DNet-Lite FULL-BANDS CV Histories ==="

cd $SLURM_SUBMIT_DIR

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Ensure Python can see your project modules
export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 experiments/hyper3dnetlite/analyze_lite_full_band_histories.py

echo "=== Done ==="
