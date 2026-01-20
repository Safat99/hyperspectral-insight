#!/bin/bash
#SBATCH --job-name=conv3d_ssep
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

#SBATCH --array=7
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

echo "===== Conv3D SSEP CV Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
NUM_BANDS=(5 20 50)

dataset_id=$(( SLURM_ARRAY_TASK_ID / 3 ))
band_id=$(( SLURM_ARRAY_TASK_ID % 3 ))

DATASET=${DATASETS[$dataset_id]}
NB=${NUM_BANDS[$band_id]}

echo "Running dataset: $DATASET"
echo "Num bands: $NB"
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
python3 experiments/conv3d_full/run_3dcnn_ssep.py \
    --dataset "$DATASET" \
    --patch 25 \
    --splits 5 \
    --epochs 50 \
    --max_samples 2000 \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --num_bands "$NB"
