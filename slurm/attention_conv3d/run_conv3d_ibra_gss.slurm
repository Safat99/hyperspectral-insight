#!/bin/bash
#SBATCH --job-name=conv3d_ibra_gss
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

#SBATCH --array=0-14
#SBATCH --output=slurm_logs/conv3d/conv3d_ibra_gss/new/%x-%A_%a.out
#SBATCH --error=slurm_logs/conv3d/conv3d_ibra_gss/new/%x-%A_%a.err

echo "===== Conv3D Full IBRA+GSS (Best Optuna Settings) Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
NUM_BANDS=(5 20 50)

dataset_id=$(( SLURM_ARRAY_TASK_ID / 3 ))
band_id=$(( SLURM_ARRAY_TASK_ID % 3 ))

DATASET=${DATASETS[$dataset_id]}
NB=${NUM_BANDS[$band_id]}

echo "Running dataset: $DATASET"
echo "Num bands: $NB"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"

# ---------------- Environment ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# =========================================================
#        DATASET-SPECIFIC BEST HYPERPARAMETERS
# =========================================================

OPTIMIZER="adam"

if [ "$DATASET" = "indian_pines" ]; then
    PATCH=17
    BATCH=8
    LR=0.0006

elif [ "$DATASET" = "salinas" ]; then
    PATCH=17
    BATCH=8
    LR=0.0005

elif [ "$DATASET" = "ksc" ]; then
    PATCH=50
    BATCH=16
    LR=0.0010

elif [ "$DATASET" = "pavia_university" ]; then
    PATCH=25
    BATCH=8
    LR=0.0010

elif [ "$DATASET" = "pavia_centre" ]; then
    PATCH=13
    BATCH=8
    LR=0.0009
fi

echo "Patch size: $PATCH"
echo "Batch size: $BATCH"
echo "Learning rate: $LR"
echo "Optimizer: $OPTIMIZER"

# =========================================================
#                    RUN EXPERIMENT
# =========================================================

python3 experiments/conv3d_full/run_3dcnn_srpa.py \
    --dataset "$DATASET" \
    --patch $PATCH \
    --splits 10 \
    --epochs 30 \
    --batch_size $BATCH \
    --learning_rate $LR \
    --optimizer $OPTIMIZER \
    --max_samples 2000 \
    --num_bands "$NB"
