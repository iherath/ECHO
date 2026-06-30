#!/usr/bin/env bash
# Submit all GenLGSM benchmark jobs to Bridges2.
# Run from /ocean/projects/cis250184p/iherath/ECHO
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../logs"

# task -> number of runs (seeds 1..N)
declare -A RUNS=(
    [diam]=1
    [ecc]=2
    [sssp]=3
    [energy]=3
    [charge]=3
)

for TASK in diam ecc sssp energy charge; do
    N=${RUNS[$TASK]}
    for RUN in $(seq 1 "$N"); do
        JOB_NAME="glgsm_${TASK}_run${RUN}"
        sbatch \
            --job-name="$JOB_NAME" \
            --export=ALL,TASK="$TASK",SEED="$RUN",RUN="$RUN" \
            "$SCRIPT_DIR/slurm_glgsm.sh"
        echo "Submitted: $JOB_NAME  (seed=$RUN)"
    done
done
