#!/bin/bash
#SBATCH --output=slurm_logs/test_submit-%j.out
#SBATCH --partition=test
#SBATCH --time=00:05:00

echo "SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"

