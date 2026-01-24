#!/bin/bash
#SBATCH --job-name=tune_h3dnet_lite
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

#SBATCH --array=0-18
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Hyper3DNet Tuning Job Started ====="
echo "Dataset: $1"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Working Dir: $SLURM_SUBMIT_DIR"

DATASET=$1
PHASE=$2   # 1 or 2

module load CUDA/12.2.0
module load Anaconda3/2024.02-1

# Activate your TF environment
source ~/.bashrc
conda activate hsi_tf

# Ensure package import works
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

mkdir -p slurm_logs


echo "Running Hyper3DNetLite tuning..."

echo "Dataset: $DATASET"
echo "Config ID: $SLURM_ARRAY_TASK_ID"
echo "Phase: $PHASE"



python experiments/hyper3dnetlite/tune_lite.py \
    --dataset "$DATASET" \
    --config_id "$SLURM_ARRAY_TASK_ID" \
    --phase "$PHASE" \
    --splits 5 \
    --epochs 10

echo "===== Hyper3DNet Tuning Completed ====="
