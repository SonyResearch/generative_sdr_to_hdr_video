#!/bin/bash
#SBATCH --job-name=sdr2hdr_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --qos=gpu1-32h
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#
# Runs single-video HDR inference on a single GPU via slurm.
# Usage: sbatch inference_slurm.sbatch [input_dir] [output_dir] [config_file]

set -x -e

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "$SCRIPT_DIR/slurm_logs"

input_dir="${1:-$SCRIPT_DIR/assets/demo_input}"
output_dir="${2:-$SCRIPT_DIR/output/demo_output}"
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

echo "START TIME: $(date)"
echo "Node: $(hostname), GPUs visible: $CUDA_VISIBLE_DEVICES"

python inference.py --input_dir "$input_dir" --output_dir "$output_dir" --config "$config_file"

echo "END TIME: $(date)"
