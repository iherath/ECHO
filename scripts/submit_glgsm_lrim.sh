#!/usr/bin/env bash
# Submit GenLGSM jobs on the LRIM-16 hard benchmark (Long-Range Ising Model) to Bridges-2.
# Uses the same slurm_glgsm.sh launcher as the other GenLGSM tasks.
# Run from /ocean/projects/cis250184p/iherath/ECHO
#
# LRIM-16 hard = lrim_16_0.6_10k (16x16 grid, sigma=0.6, 10k graphs), node-level energy MSE.
# Config matches the LGSM baseline from Table 8 + Table 10 of the LGSM paper (target: logMSE
# -4.284 +/- 0.133 on LRIM-16-hard):
#   num_steps (sequence length) = 32, num_layers (blocks) = 8, hidden_dim = 64 (from slurm_glgsm.sh),
#   batch_size = 32, lr = 3e-4.
# We run glgsm_mode=hyper (our mechanism) instead of LGSM's NBT recurrence, everything else matched.
# Single GPU (1 node, batch 32) to stay in the paper's batch-32 regime -- the sbatch overrides below
# force 1 GPU so effective batch stays 32 (16-GPU DDP would make it 512 and need lr rescaling).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../logs"

# which LRIM file to pull from HuggingFace (16-grid, sigma=0.6 hard, 10k graphs)
export LRIM_NAME="lrim_16_0.6_10k"

# Format: "task runs num_steps num_layers lr max_epochs seed_start"
TASK_CONFIGS=(
    "lrim 3  32  8  0.0003  300  1"
)

HYPER_HIDDEN=128
WINDOW_SIZE=2

for config in "${TASK_CONFIGS[@]}"; do
    read -r TASK N NUM_STEPS NUM_LAYERS LR MAX_EPOCHS SEED_START <<< "$config"
    for RUN in $(seq "$SEED_START" $((SEED_START + N - 1))); do
        JOB_NAME="glgsm_${TASK}_run${RUN}"
        sbatch \
            --job-name="$JOB_NAME" \
            --nodes=1 --ntasks-per-node=1 --gpus=v100-32:1 \
            --export=ALL,TASK="$TASK",LRIM_NAME="$LRIM_NAME",SEED="$RUN",RUN="$RUN",NUM_STEPS="$NUM_STEPS",NUM_LAYERS="$NUM_LAYERS",LR="$LR",MAX_EPOCHS="$MAX_EPOCHS",HYPER_HIDDEN="$HYPER_HIDDEN",WINDOW_SIZE="$WINDOW_SIZE" \
            "$SCRIPT_DIR/slurm_glgsm.sh"
        echo "Submitted: $JOB_NAME  (steps=$NUM_STEPS layers=$NUM_LAYERS lr=$LR max_epochs=$MAX_EPOCHS seed=$RUN)"
    done
done
