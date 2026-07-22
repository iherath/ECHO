#!/usr/bin/env bash
# Submit only the GenLGSM charge and energy ECHO benchmark jobs to Bridges-2.
# Same hyperparams and resources (via slurm_glgsm.sh) as submit_glgsm.sh; use
# this to (re)run just the chem tasks after the scaling_factor fix in train.py.
# Run from /ocean/projects/cis250184p/iherath/ECHO
#
# Hyperparams per task (Table 8, LGSM paper):
#   energy    : lr=3e-4, steps=32, layers=2
#   charge    : lr=3e-4, steps=40, layers=4
#
# Multi-node DDP: slurm_glgsm.sh runs 16 V100s (2 GPU-partition nodes), so effective
# batch = 16 x 32 = 512 (was 32). lr below is the paper value at batch 32; at this batch
# scale lr up (sqrt rule ~1.2e-3, or linear ~4.8e-3 + warmup) or convergence will degrade.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../logs"

# Format: "task runs num_steps num_layers lr"
TASK_CONFIGS=(
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
