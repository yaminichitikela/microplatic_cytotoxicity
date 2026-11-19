#!/bin/bash
#SBATCH --job-name=viab_cv_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --partition=main

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/cv_report.py
