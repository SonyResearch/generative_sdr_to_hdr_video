#!/bin/bash

# Environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=7
export OPENCV_IO_ENABLE_OPENEXR=1

# Training command
accelerate launch examples/wanvideo/model_training/train_decoder.py --config $SCRIPT_DIR/diffsynth/configs/finetune_decoder_justmerger_stage2ea.yaml