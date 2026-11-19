#!/bin/bash
# Runs line (SLURM_ARRAY_TASK_ID + 1) of the task file $TASK_FILE: "<script> <args...>"
#SBATCH --job-name=viab_cv
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# TODO: set these for your Wahab account if required (check with your group / `sinfo`)
##SBATCH --partition=main
##SBATCH --account=<your_account>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
echo "task $SLURM_ARRAY_TASK_ID: $LINE  (cpus=$SLURM_CPUS_PER_TASK)"
python $LINE
