#!/bin/bash
#SBATCH --job-name=h3dnet_full
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Hyper3DNet FULL-BAND CV Started ====="
# echo "Dataset: $1"
DATASETS=("salinas" "pavia_university" "pavia_centre")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Submit Dir: $SLURM_SUBMIT_DIR"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/hyper3dnet/run_hyper3dnet_original_cv.py \
    --dataset "$DATASET" \
    --patch 25 \
    --splits 10 \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_samples 2000

echo "===== Hyper3DNet FULL Completed ====="
