#!/bin/bash
# Single GenLGSM training job for Bridges-2.
# All caps vars are injected by submit_glgsm.sh via --export.
#SBATCH -A cis250184p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH -t 48:00:00
#SBATCH --output=logs/glgsm_%x_%j.out
#SBATCH --error=logs/glgsm_%x_%j.err

set -euo pipefail

CONDA_ENV=/ocean/projects/cis250184p/iherath/conda_envs/graph-ssm
ECHO_DIR=/ocean/projects/cis250184p/iherath/ECHO

module load cuda/12.4.0

source /jet/home/iherath/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$ECHO_DIR"

echo "=== GenLGSM  task=${TASK}  seed=${SEED}  run=${RUN}  node=$(hostname) ==="
echo "    num_steps=${NUM_STEPS}  num_layers=${NUM_LAYERS}  lr=${LR}"

python scripts/train.py \
    --task             "$TASK" \
    --seed             "$SEED" \
    --gnn_type         GenLGSM \
    --glgsm_mode       hyper \
    --hidden_dim       64 \
    --num_layers       "$NUM_LAYERS" \
    --d_state          64 \
    --hyper_hidden_dim "$HYPER_HIDDEN" \
    --window_size      "$WINDOW_SIZE" \
    --num_steps        "$NUM_STEPS" \
    --lr               "$LR" \
    --batch_size       32 \
    --max_epochs       1000 \
    --es_patience      100 \
    --dropout          0.0 \
    --weight_decay     0.0 \
    --lr_scheduler     none \
    --device           gpu \
    --wandb
