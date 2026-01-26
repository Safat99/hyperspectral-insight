#!/bin/bash
#SBATCH --job-name=run_lite_ibra_gss
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00

#SBATCH --array=0-14
#SBATCH --output=slurm_logs/lite/new/%x-%A_%a.out
#SBATCH --error=slurm_logs/lite/new/%x-%A_%a.err

echo "===== Hyper3DNet-Lite IBRA+GSS Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
NUM_BANDS_LIST=(5 20 50)

dataset_id=$(( SLURM_ARRAY_TASK_ID / 3 ))
bands_id=$(( SLURM_ARRAY_TASK_ID % 3 ))

DATASET=${DATASETS[$dataset_id]}
NUM_BANDS=${NUM_BANDS_LIST[$bands_id]}

echo "Dataset: $DATASET"
echo "Bands: $NUM_BANDS"
echo "Node: $(hostname)"

# ---------------- Environment ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# =========================================================
#        DATASET-SPECIFIC TUNED HYPERPARAMETERS
# =========================================================

if [ "$DATASET" = "indian_pines" ]; then
    PATCH=17
    BATCH=32
    OPTIMIZER="adam"
    LR=1.02e-4

elif [ "$DATASET" = "ksc" ]; then
    PATCH=17
    BATCH=8
    OPTIMIZER="adam"
    LR=3.22e-4

elif [ "$DATASET" = "pavia_centre" ]; then
    PATCH=25
    BATCH=64
    OPTIMIZER="adam"
    LR=2.10e-4

elif [ "$DATASET" = "pavia_university" ]; then
    PATCH=9
    BATCH=16
    OPTIMIZER="adam"
    LR=1.04e-4

elif [ "$DATASET" = "salinas" ]; then
    PATCH=17
    BATCH=32
    OPTIMIZER="adadelta"
    RHO=0.9705
    EPS=2.78e-4
fi

echo "Patch size: $PATCH"
echo "Batch size: $BATCH"
echo "Optimizer: $OPTIMIZER"

# =========================================================
#                    RUN EXPERIMENT
# =========================================================

if [ "$OPTIMIZER" = "adam" ]; then
    python experiments/hyper3dnetlite/run_lite_ibra_gss.py \
        --dataset "$DATASET" \
        --num_bands "$NUM_BANDS" \
        --patch $PATCH \
        --splits 10 \
        --epochs 25 \
        --batch_size $BATCH \
        --learning_rate $LR \
        --max_samples 2000

else
    python experiments/hyper3dnetlite/run_lite_ibra_gss.py \
        --dataset "$DATASET" \
        --num_bands "$NUM_BANDS" \
        --patch $PATCH \
        --splits 10 \
        --epochs 25 \
        --batch_size $BATCH \
        --learning_rate 1.0 \  # dummy, not used
        --max_samples 2000 \
        --optimizer adadelta \
        --rho $RHO \
        --epsilon $EPS
fi
