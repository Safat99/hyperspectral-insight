#!/bin/bash
#SBATCH --job-name=ssl_lite_ibra_gss
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --array=0-4
#SBATCH --output=slurm_logs/ssl/new/%x-%A_%a.out
#SBATCH --error=slurm_logs/ssl/new/%x-%A_%a.err

echo "===== SSL Scaling (Hyper3DNet-Lite + IBRA+GSS) Started ====="

DATASETS=("indian_pines" "salinas" "pavia_university" "pavia_centre" "ksc")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Dataset: $DATASET"
echo "Node: $(hostname)"

# ---------------- Environment ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/ssl/new

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
#                    SSL SETTINGS
# =========================================================
SSL_ITERS=3
EPOCHS_BASELINE=30
CONF_THRESH=0.9
MAX_SAMPLES=2000
N_SPLITS=10

# Epochs per SSL iteration (to keep total training budget equal)
EPOCHS_PER_ITER=$((EPOCHS_BASELINE / SSL_ITERS))

echo "SSL Iterations: $SSL_ITERS"
echo "Epochs per SSL iteration: $EPOCHS_PER_ITER"
echo "Total SSL epochs: $((EPOCHS_PER_ITER * SSL_ITERS))"

# =========================================================
#                    RUN SSL EXPERIMENT
# =========================================================

if [ "$OPTIMIZER" = "adam" ]; then
    python experiments/ssl/run_ssl_scale_experiment.py \
        --dataset "$DATASET" \
        --patch_size $PATCH \
        --batch_size $BATCH \
        --learning_rate $LR \
        --n_splits $N_SPLITS \
        --ssl_iters $SSL_ITERS \
        --epochs_baseline $EPOCHS_BASELINE \
        --confidence_thresh $CONF_THRESH \
        --max_samples $MAX_SAMPLES

else
    python experiments/ssl/run_ssl_scale_experiment.py \
        --dataset "$DATASET" \
        --patch_size $PATCH \
        --batch_size $BATCH \
        --optimizer adadelta \
        --rho $RHO \
        --epsilon $EPS \
        --n_splits $N_SPLITS \
        --ssl_iters $SSL_ITERS \
        --epochs_baseline $EPOCHS_BASELINE \
        --confidence_thresh $CONF_THRESH \
        --max_samples $MAX_SAMPLES
fi

echo "===== SSL Scaling Experiment Completed ====="
