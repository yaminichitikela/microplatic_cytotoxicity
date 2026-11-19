#!/bin/bash
# Single random-split runs (src/train.py), then the report. The main evaluation is nested CV: ./submit_cv.sh
# Usage:  ./submit.sh                                   features C, seed 42 -> 4 tasks
#         repeated random splits (10 repeats x 5 feature sets x 4 models = 200 tasks):
#         FEATURE_SETS="C B E6 E5 E4" N_SEEDS=10 FIRST_SEED=0 ./submit.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

FEATURE_SETS=${FEATURE_SETS:-C}
N_SEEDS=${N_SEEDS:-1}
FIRST_SEED=${FIRST_SEED:-42}
N_TRIALS=${N_TRIALS:-100}
read -r -a F <<< "$FEATURE_SETS"
N_TASKS=$((N_SEEDS * ${#F[@]} * 4))

export FEATURE_SETS N_SEEDS FIRST_SEED N_TRIALS
ARRAY_ID=$(sbatch --parsable --array=0-$((N_TASKS - 1)) --export=ALL slurm/tune_array.sbatch)
REPORT_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" slurm/report.sbatch)
echo "[$FEATURE_SETS]: tuning array $ARRAY_ID ($N_TASKS tasks)   report: $REPORT_ID"
echo "watch with:   squeue -u $USER"
