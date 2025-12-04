#!/bin/bash
#SBATCH --job-name=ssl_plot
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== SSL Plotting Started ====="

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 plot_ssl_scaling_results.py \
    --json results/scaling_ssl/"$1"_ssl_scaling_results.json \
    --save results/scaling_ssl/"$1"_ssl_scaling_plot.png

echo "===== SSL Plotting Completed ====="
