#!/bin/bash

# Check if the checkpoint path is provided as an argument
# pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

CHECKPOINT_PATH=$1

vllm serve $CHECKPOINT_PATH \
    --port 8000 \
    --served-model-name "qwen2vl" \
    --gpu-memory-utilization 0.6 \
    --uvicorn-log-level info \
    --api-key "ical" 2>&1 | tee "vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi
