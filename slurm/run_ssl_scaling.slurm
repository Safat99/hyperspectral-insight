#!/bin/bash
#SBATCH --job-name=ssl_scaling
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --array=0-4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== SSL Scaling Experiment Started ====="
DATASETS=("salinas" "pavia_university" "pavia_centre" "ksc", "indian_pines")
# DATASETS=("salinas" "pavia_centre")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/ssl/run_ssl_scaling_experiment.py \
    --dataset "$DATASET" \
    --max_samples 2000 \
    --batch_size 128 \
    --learning_rate 5e-4

echo "===== SSL Scaling Experiment Completed ====="
