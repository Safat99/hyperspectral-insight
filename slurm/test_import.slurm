#!/bin/bash
#SBATCH --output=slurm_logs/test-%j.out
#SBATCH --error=slurm_logs/test-%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=test

export PYTHONPATH="$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate hsi

python3 - << 'EOF'
import sys
print("SLURM SUBMIT DIR:", "$SLURM_SUBMIT_DIR")
print("PYTHONPATH =", sys.path)

try:
    from hyperspectral_insight.data.loaders import load_dataset
    print("Import SUCCESS")
except Exception as e:
    print("Import FAILED:", e)
EOF

