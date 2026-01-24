#!/bin/bash
#SBATCH --job-name=analyze_h3dnet
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

echo "=== Plotting hyper3dnet Histories ==="
echo "Dataset: $1 Variant: $2"
cd $SLURM_SUBMIT_DIR

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

# Make sure Python can import your local package
export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 experiments/hyper3dnet/analyze_h3dnet_histories.py --dataset "$1" --variant "$2"


echo "=== Done ==="
