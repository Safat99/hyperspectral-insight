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

#SBATCH --array=0-9
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Hyper3DNet-Lite IBRA_GSS Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
NUM_BANDS_LIST=(5 50)

# -----------------------------
# Compute indices
# -----------------------------
dataset_id=$(( SLURM_ARRAY_TASK_ID / 2 ))   # 0–4
bands_id=$(( SLURM_ARRAY_TASK_ID % 2 ))     # 0–1

DATASET=${DATASETS[$dataset_id]}
NUM_BANDS=${NUM_BANDS_LIST[$bands_id]}

echo "SLURM TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Selected dataset: $DATASET"
echo "Selected number of bands: $NUM_BANDS"

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

python3 experiments/hyper3dnetlite/run_lite_ibra_gss.py \
    --dataset "$DATASET" \
    --num_bands "$NUM_BANDS" \
    --patch 25 \
    --splits 10 \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --max_samples 2000