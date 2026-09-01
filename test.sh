#!/bin/bash

# Environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_P2P_LEVEL=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

# Set CUDA_VISIBLE_DEVICES yourself beforehand to pick which GPU(s) to use.
# Defaults to a single GPU; set CUDA_VISIBLE_DEVICES to a comma-separated list
# (e.g. "0,1,2,3") to automatically fan the video-level validation loop out
# across that many GPUs.
# Usage: bash test.sh [dataset] [ae_types]
#   dataset:  stuttgart (default) or ubc
#   ae_types: comma-separated exposure modes to test (default: auto,over,under)
DATASET="${1:-stuttgart}"
AE_TYPES="${2:-auto,over,under}"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NUM_GPUS=$(($(grep -o ',' <<< "$CUDA_VISIBLE_DEVICES" | wc -l) + 1))
else
  NUM_GPUS=1
fi

cd "$SCRIPT_DIR"
accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --gpu_ids all --mixed_precision no \
  test.py --config "$SCRIPT_DIR/diffsynth/configs/threeexposures_crffixed_test_val.yaml" \
  --dataset "$DATASET" --ae-types "$AE_TYPES"
