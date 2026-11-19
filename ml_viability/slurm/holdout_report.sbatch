#!/bin/bash
#SBATCH --job-name=viab_holdout_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --partition=main

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/holdout_report.py --n-seeds "${N_SEEDS:-5}"
