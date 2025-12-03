#!/bin/bash
#SBATCH --job-name=plot_shallow_hist
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Plotting Training Histories ==="
echo "Dataset: $1"

cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

# No GPU needed, but partition must be gpupriority for your account
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 experiments/baseline/plot_shallow_history.py --dataset "$1"

echo "=== Plotting Completed ==="
