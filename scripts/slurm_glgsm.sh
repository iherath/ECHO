#!/bin/bash
#SBATCH --account=cis250184p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=12:00:00

# TASK, SEED, and RUN are injected by submit_glgsm.sh via --export
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

python scripts/train.py \
    --task        "$TASK" \
    --seed        "$SEED" \
    --gnn_type    GenLGSM \
    --glgsm_mode  hyper \
    --hidden_dim  64 \
    --num_layers  4 \
    --d_state     64 \
    --hyper_hidden_dim 128 \
    --window_size 2 \
    --lr          0.0001 \
    --batch_size  32 \
    --max_epochs  1000 \
    --es_patience 100 \
    --dropout     0.0 \
    --weight_decay 0.0 \
    --lr_scheduler none \
    --device      gpu \
    --wandb
