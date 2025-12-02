#!/bin/bash
#SBATCH --job-name=tf_gpu_test
#SBATCH --account=priority-binhaizhu
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate hsi_tf

python3 - << 'EOF'
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
EOF

