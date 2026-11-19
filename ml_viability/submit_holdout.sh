#!/bin/bash
# Repeated random split with the same models and tuning as the nested CV (all 13 features by default).
#   stage 1: base models  (models x seeds; each seed = one stratified 80/20 split)
#   stage 2: ensembles    (consensus + stacking of the top 3 per seed)       after stage 1
#   stage 3: report       (results_holdout/summary.csv)                      after stage 2
# Usage:  ./submit_holdout.sh                 5 repeats -> 30 model tasks + 5 ensemble tasks
#         N_SEEDS=10 ./submit_holdout.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs tasks

FEATURES=${FEATURES:-C}
MODELS=${MODELS:-"xgb rf et svr knn mlp"}
N_SEEDS=${N_SEEDS:-5}
N_TRIALS=${N_TRIALS:-50}

: > tasks/holdout_base.txt; : > tasks/holdout_ensemble.txt
for s in $(seq 0 $((N_SEEDS - 1))); do
  for m in $MODELS; do
    echo "src/cv_train.py --scheme holdout --features $FEATURES --model $m --seed $s --fold 0 --n-trials $N_TRIALS" >> tasks/holdout_base.txt
  done
  echo "src/cv_ensemble.py --scheme holdout --features $FEATURES --seed $s --fold 0" >> tasks/holdout_ensemble.txt
done

export N_SEEDS
NB=$(wc -l < tasks/holdout_base.txt); NE=$(wc -l < tasks/holdout_ensemble.txt)
B=$(sbatch --parsable --array=0-$((NB - 1)) --export=ALL,TASK_FILE=tasks/holdout_base.txt slurm/cv_task.sbatch)
E=$(sbatch --parsable --array=0-$((NE - 1)) --dependency=afterany:"$B" \
      --export=ALL,TASK_FILE=tasks/holdout_ensemble.txt slurm/cv_task.sbatch)
R=$(sbatch --parsable --dependency=afterany:"$E" --export=ALL slurm/holdout_report.sbatch)
echo "base models: $B ($NB tasks)   ensembles: $E ($NE tasks)   report: $R"
echo "watch with:  squeue -u $USER"
