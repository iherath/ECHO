#!/bin/bash
# Multi-node DDP smoke test on the full energy setup. Verifies end-to-end:
#   1. 16 ranks across 2 GPU-partition nodes register with NCCL over InfiniBand,
#   2. one full epoch completes -- backward all-reduce through the custom GenLGSM/mamba
#      layers, end-of-epoch val/test/checkpoint, and the rank-0 summary print.
# 16 GPUs is ~4x the timed-out 4-GPU smoke, so the energy epoch should finish well inside
# the 1h cap (SLURM bills only actual runtime, so a roomy cap is free insurance).
# Run from ECHO_DIR (ensure logs/ exists):  sbatch scripts/slurm_glgsm_smoke.sh
#SBATCH -A cis250184p
#SBATCH -p GPU
#SBATCH -N 2
#SBATCH --gpus=v100-32:16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH -t 01:00:00
#SBATCH --job-name=glgsm_smoke
#SBATCH --output=logs/glgsm_%x_%j.out
#SBATCH --error=logs/glgsm_%x_%j.err

set -euo pipefail

CONDA_ENV=/ocean/projects/cis250184p/iherath/conda_envs/graph-ssm
ECHO_DIR=/ocean/projects/cis250184p/iherath/ECHO

module load cuda/12.4.0

source /jet/home/iherath/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$ECHO_DIR"

echo "=== GenLGSM DDP smoke  nodes=${SLURM_NNODES}  gpus/node=${SLURM_NTASKS_PER_NODE} ==="

# no --wandb: use the CSV logger so the smoke test needs no wandb login
srun python scripts/train.py \
    --task             energy \
    --seed             1 \
    --gnn_type         GenLGSM \
    --glgsm_mode       hyper \
    --hidden_dim       64 \
    --num_layers       2 \
    --d_state          64 \
    --hyper_hidden_dim 128 \
    --window_size      2 \
    --num_steps        32 \
    --lr               0.0003 \
    --batch_size       32 \
    --max_epochs       1 \
    --es_patience      100 \
    --dropout          0.0 \
    --weight_decay     0.0 \
    --lr_scheduler     none \
    --device           gpu \
    --devices          "${SLURM_NTASKS_PER_NODE}" \
    --num_nodes        "${SLURM_NNODES}"
