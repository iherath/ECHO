#!/bin/bash
# Quick DDP smoke test: 1 epoch of the energy task across 4 V100s to confirm all
# ranks initialize and an epoch completes before committing a full 48h run.
# Run from ECHO_DIR (ensure logs/ exists):  sbatch scripts/slurm_glgsm_smoke.sh
#SBATCH -A cis250184p
#SBATCH -p GPU-shared
#SBATCH --qos=gpu
#SBATCH -N 1
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH -t 00:20:00
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

echo "=== GenLGSM DDP smoke  node=$(hostname)  gpus=${SLURM_NTASKS_PER_NODE} ==="

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
    --devices          "${SLURM_NTASKS_PER_NODE}"
