#!/bin/bash

# Environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_P2P_LEVEL=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

# Set CUDA_VISIBLE_DEVICES yourself beforehand to pick a GPU (defaults to the first visible one).
cd "$SCRIPT_DIR"
accelerate launch --num_processes 1 --num_machines 1 --gpu_ids all --mixed_precision no \
  examples/wanvideo/model_training/train.py --config "$SCRIPT_DIR/diffsynth/configs/threeexposures_crffixed_test_val.yaml"