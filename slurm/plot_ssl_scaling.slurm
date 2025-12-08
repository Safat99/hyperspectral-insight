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

DATASETS=("salinas" "pavia_university" "pavia_centre" "ksc" "indian_pines")
# DATASETS=("salinas" "pavia_centre")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/ssl/plot_ssl_scaling_result.py \
    --json results/scaling_ssl/updated/"$DATASET"_ssl_b128_n5_ssl_scaling_results.json \
    --save results/scaling_ssl/updated/"$DATASET"_ssl_b128_n5_scaling_plot.png

echo "===== SSL Plotting Completed ====="
