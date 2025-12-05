#!/bin/bash
#SBATCH --job-name=h3dnet_lite_ibra_gss
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

#SBATCH --array=0-4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Hyper3DNet-Lite FULL-BAND CV Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
# DATASETS=("salinas" "pavia_centre")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running dataset: $DATASET"
echo "Array task: $SLURM_ARRAY_TASK_ID"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# -----------------------------
# Run Hyper3DNet-Lite CV
# -----------------------------
python3 experiments/hyper3dnetlite/run_lite_ibra_gss.py \
    --dataset "$DATASET" \
    --patch 25 \
    --splits 5 \
    --epochs 50 \
    --max_samples 2000 \
    --batch_size 128 \
    --learning_rate 5e-4
