#!/bin/bash
#SBATCH --job-name=tune_h3dnet
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "===== Hyper3DNet Tuning Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Working Dir: $SLURM_SUBMIT_DIR"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1

# --- Add this line to suppress the cuda_timer warning ---
export TF_GPU_THREAD_COUNT=1

# Activate your TF environment
source ~/.bashrc
conda activate hsi_tf

# Ensure package import works
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

mkdir -p slurm_logs

echo "Running Hyper3DNet tuning..."

python3 experiments/hyper3dnet/tune_hyper3dnet.py \
    --dataset indian_pines \
    --splits 3 \
    --epochs 20

echo "===== Hyper3DNet Tuning Completed ====="
