#!/bin/bash
#SBATCH --job-name=select_best_hparams
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

echo "Running best-hyperparameter extraction..."
python3 experiments/conv3d_full/select_best_hparams.py

echo "Submitting next job..."

jid3=$(sbatch generated_slurm/conv3d_full_best.slurm | awk '{print $4}')

echo "Submitted final model run with Job ID: $jid3"
