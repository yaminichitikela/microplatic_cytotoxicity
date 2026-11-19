#!/bin/bash
#SBATCH --job-name=viab_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --partition=main

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/report.py
if [ "${N_SEEDS:-1}" -gt 1 ]; then
  python src/repeated_report.py --n-seeds "$N_SEEDS" --first-seed "${FIRST_SEED:-0}"
fi
