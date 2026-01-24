#!/bin/bash
#SBATCH --account=priority-binhaizhu       # Your PI's priority account
#SBATCH --job-name=ssl_kfold
#SBATCH --partition=gpupriority            # Correct GPU partition on Tempest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                  # Do not exceed 16 cores per GPU
#SBATCH --gpus-per-task=1                  # 1 A40 GPU
#SBATCH --mem=60000                        # 60 GB RAM
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl/kfold_%A_%a.out
#SBATCH --error=logs/ssl/kfold_%A_%a.err
#SBATCH --array=0-2                        # datasets: IP, Salinas, Pavia

# --------------------------
# Dataset list for array
# --------------------------
DATASETS=("indian_pines" "salinas" "pavia")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running SSL K-Fold for dataset: $DATASET"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# --------------------------
# Load modules
# --------------------------
module load Anaconda3/2020.07
module load CUDA/11.1.1-GCC-10.2.0

# --------------------------
# Activate Conda environment
# --------------------------
source activate hyperspectral-insight

# --------------------------
# Ensure log directory exists
# --------------------------
mkdir -p logs/ssl/

# --------------------------
# Run experiment
# --------------------------
python3 experiments/ssl/run_ssl_lite_ibra_gss_kfold.py \
    --dataset $DATASET \
    --num_bands 20 \
    --patch 25 \
    --splits 10 \
    --ssl_iters 5 \
    --epochs 20 \
    --batch_size 16 \
    --conf 0.90

echo "Finished SSL K-fold for $DATASET"
