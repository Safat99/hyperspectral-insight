#!/bin/bash
#SBATCH --job-name=shallow_baseline
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Starting Shallow CNN Baseline ==="
echo "Dataset: $1"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Submit Dir: $SLURM_SUBMIT_DIR"

# Go to working directory
cd $SLURM_SUBMIT_DIR
mkdir -p slurm_logs

# Load CUDA 12.2 for tensorflow 2.20
module load CUDA/12.2.0

# Load Anaconda and activate your env
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Make sure Python can import your local package
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Python version: $(python3 --version)"
echo "Running baseline for dataset = $1"

python3 experiments/baseline/run_shallow_cnn.py --dataset "$1"
# python3 experiments/baseline/run_shallow_cnn.py \
    # --dataset pavia_centre \
    # --patch 25 \
    # --max_samples 2000

echo "=== Job Completed ==="

