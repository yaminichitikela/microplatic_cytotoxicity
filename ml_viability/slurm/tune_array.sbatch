#!/bin/bash
# Single random-split runs (src/train.py): one array task per (seed x feature set x model).
# The array size is set by submit.sh: N_SEEDS x (number of FEATURE_SETS) x 4 models.
#SBATCH --job-name=viab_tune
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# TODO: set these for your Wahab account if required (check with your group / `sinfo`)
##SBATCH --partition=main
##SBATCH --account=<your_account>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

read -r -a FEATURES <<< "${FEATURE_SETS:-C}"
MODELS=(rf svm xgb knn)
N_SEEDS=${N_SEEDS:-1}
FIRST_SEED=${FIRST_SEED:-42}
N_TRIALS=${N_TRIALS:-100}
NF=${#FEATURES[@]}

i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((i % 4))]}
FEAT=${FEATURES[$((i / 4 % NF))]}
SEED=$((FIRST_SEED + i / (4 * NF) % N_SEEDS))

echo "task $i: seed=$SEED features=$FEAT model=$MODEL trials=$N_TRIALS cpus=$SLURM_CPUS_PER_TASK"
EXTRA=""
if [ "$N_SEEDS" -gt 1 ]; then EXTRA="--no-save-model"; fi   # repeated splits: skip model files (disk space)
python src/train.py --model "$MODEL" --features "$FEAT" --seed "$SEED" --n-trials "$N_TRIALS" $EXTRA
