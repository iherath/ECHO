#!/usr/bin/env bash
# Submit all GenLGSM ECHO benchmark jobs to Bridges-2.
# Run from /ocean/projects/cis250184p/iherath/ECHO
#
# Hyperparams per task (Table 8, LGSM paper):
#   diam/ecc  : hparams.yaml takes priority → lr=1e-4, steps=40, layers=4
#   sssp      : Table 8                     → lr=3e-4, steps=40, layers=4
#   energy    : Table 8                     → lr=3e-4, steps=32, layers=2
#   charge    : Table 8                     → lr=3e-4, steps=40, layers=4
#
# Conflict (diam & ecc): Table 8 says lr=3e-4; hparams.yaml says lr=1e-4 → using 1e-4
#
# Note: Table 8 says "none" normalization for sssp and energy, but the
# GenLGSM code always uses the normalized (row-stochastic) path when
# max_hops > 20. Both tasks exceed this threshold so they run normalized.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../logs"

# Format: "task runs num_steps num_layers lr"
TASK_CONFIGS=(
    "diam   1  40  4  0.0001"
    "ecc    2  40  4  0.0001"
    "sssp   3  40  4  0.0003"
    "energy 3  32  2  0.0003"
    "charge 3  40  4  0.0003"
)

# hyper_hidden_dim and window_size not specified in Table 8; use best known values from hparams
HYPER_HIDDEN=128
WINDOW_SIZE=2

for config in "${TASK_CONFIGS[@]}"; do
    read -r TASK N NUM_STEPS NUM_LAYERS LR <<< "$config"
    for RUN in $(seq 1 "$N"); do
        JOB_NAME="glgsm_${TASK}_run${RUN}"
        sbatch \
            --job-name="$JOB_NAME" \
            --export=ALL,TASK="$TASK",SEED="$RUN",RUN="$RUN",NUM_STEPS="$NUM_STEPS",NUM_LAYERS="$NUM_LAYERS",LR="$LR",HYPER_HIDDEN="$HYPER_HIDDEN",WINDOW_SIZE="$WINDOW_SIZE" \
            "$SCRIPT_DIR/slurm_glgsm.sh"
        echo "Submitted: $JOB_NAME  (steps=$NUM_STEPS layers=$NUM_LAYERS lr=$LR seed=$RUN)"
    done
done
