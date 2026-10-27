#!/bin/bash
# Train a DVAE on one Slurm node with every GPU it was given (one torchrun process per GPU).
#
#   sbatch --account=<your-account> scripts/slurm_train.sh data/notes runs/k5 [dvae-train options]
#
# Arguments: training sounds (a folder, a file list, or a .tar of a folder), the output folder,
# then any further dvae-train options. A .tar is unpacked to $SLURM_TMPDIR first, which is much
# faster than reading many small files from a network file system.
#
# Environment: VENV, a virtual environment with this package installed (default: .venv).
# Resubmit with `--resume <out>/resume.pt` to continue a run that hit the time limit.
#
#SBATCH --job-name=dvae-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: sbatch scripts/slurm_train.sh TRAIN_DATA OUT_DIR [dvae-train options]" >&2
    exit 1
fi
train_data=$1
out=$2
shift 2

source "${VENV:-.venv}/bin/activate"

if [[ $train_data == *.tar ]]; then
    tmp=$(mktemp -d -p "${SLURM_TMPDIR:-/tmp}")
    echo "Unpacking $train_data to $tmp"
    tar -xf "$train_data" -C "$tmp"
    train_data=$tmp  # searched recursively
fi

n_gpus=$(nvidia-smi --list-gpus | wc -l)
workers=$(( ${SLURM_CPUS_PER_TASK:-4} / n_gpus ))
export OMP_NUM_THREADS=1

echo "Training on $n_gpus GPU(s), $workers data-loader worker(s) per GPU"
torchrun --standalone --nproc_per_node="$n_gpus" -m dvae_bss.cli.train \
    --train-data "$train_data" --out "$out" --num-workers "$workers" "$@"
