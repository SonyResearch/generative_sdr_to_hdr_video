#!/bin/bash
#SBATCH --job-name=sdr2hdr_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --qos=gpu1-32h
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#
# Runs the held-out validation/test pipeline on a single GPU via slurm.
# Usage: sbatch test_slurm.sbatch [dataset] [ae_types] [config_file]
#   dataset:  stuttgart (default) or ubc
#   ae_types: comma-separated exposure modes to test (default: auto,over,under)

set -x -e

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "$SCRIPT_DIR/slurm_logs"

dataset="${1:-stuttgart}"
ae_types="${2:-auto,over,under}"
config_file="${3:-$SCRIPT_DIR/diffsynth/configs/threeexposures_crffixed_test_val.yaml}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_P2P_LEVEL=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

# Defaults to 1 GPU (matches the #SBATCH --gpus-per-node above). Request more
# at submit time (e.g. `sbatch --gpus-per-node=4 test_slurm.sbatch ...`) and
# this automatically fans the video-level validation loop out across all of them.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

echo "START TIME: $(date)"
echo "Node: $(hostname), GPUs visible: $CUDA_VISIBLE_DEVICES, using $NUM_GPUS process(es)"

accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --gpu_ids all --mixed_precision no \
  test.py --config "$config_file" --dataset "$dataset" --ae-types "$ae_types"

echo "END TIME: $(date)"
