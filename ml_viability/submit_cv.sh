#!/bin/bash
# Nested cross-validation with random stratified folds on model_data.
#   stage 1: base models   (feature sets x models x seeds x 5 outer folds) + y-scrambling (features C, rf + xgb, seed 0)
#   stage 2: ensembles     (consensus + stacking per feature set x seed x fold)   after stage 1
#   stage 3: report                                                              after stage 2
# Usage:  ./submit_cv.sh                  (defaults below)
#         N_SEEDS=5 N_TRIALS=100 ./submit_cv.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs tasks

FEATURE_SETS=${FEATURE_SETS:-"B C"}
MODELS=${MODELS:-"enet pls knn svr rf et xgb mlp"}
N_SEEDS=${N_SEEDS:-3}
N_TRIALS=${N_TRIALS:-50}

: > tasks/base.txt; : > tasks/ensemble.txt
for f in $FEATURE_SETS; do for s in $(seq 0 $((N_SEEDS - 1))); do for k in 0 1 2 3 4; do
  for m in $MODELS; do
    echo "src/cv_train.py --features $f --model $m --seed $s --fold $k --n-trials $N_TRIALS" >> tasks/base.txt
  done
  echo "src/cv_ensemble.py --features $f --seed $s --fold $k" >> tasks/ensemble.txt
done; done; done
for m in rf xgb; do for k in 0 1 2 3 4; do
  echo "src/cv_train.py --features C --model $m --seed 0 --fold $k --n-trials $N_TRIALS --scramble" >> tasks/base.txt
done; done

NB=$(wc -l < tasks/base.txt); NE=$(wc -l < tasks/ensemble.txt)
B=$(sbatch --parsable --array=0-$((NB - 1)) --export=ALL,TASK_FILE=tasks/base.txt slurm/cv_task.sbatch)
E=$(sbatch --parsable --array=0-$((NE - 1)) --dependency=afterany:"$B" \
      --export=ALL,TASK_FILE=tasks/ensemble.txt slurm/cv_task.sbatch)
R=$(sbatch --parsable --dependency=afterany:"$E" slurm/cv_report.sbatch)
echo "base models: $B ($NB tasks)   ensembles: $E ($NE tasks)   report: $R"
echo "watch with:  squeue -u $USER"
