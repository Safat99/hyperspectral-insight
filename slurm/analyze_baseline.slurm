#!/bin/bash
#SBATCH --job-name=analyze_baseline
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Plotting Baseline Histories ==="
echo "Dataset: $1"

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

python3 experiments/baseline/analyze_baseline_results.py --dataset "$1"

echo "=== Done ==="