#!/bin/bash
#
# FLARE25 Evaluation Runner Script
# =================================
# This script runs evaluation on the trained Qwen3-VL model
#

set -e

# Activate UV virtual environment
VENV_PATH="/home/jma/Documents/multimodal/shuolinyin/FLARE25/.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating UV virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: UV virtual environment not found at $VENV_PATH"
    echo "Continuing with current Python environment..."
fi

# Configuration
MODEL_PATH="${MODEL_PATH:-/home/jma/Documents/multimodal/shuolinyin/FLARE25/Qwen3-VL/qwen-vl-finetune/output/qwen3vl_flare25}"
DATASET_PATH="${DATASET_PATH:-/home/jma/Documents/multimodal/shuolinyin/FLARE25/organized_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE="${DEVICE:-cuda}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"

echo "Baseline Model: $BASELINE_MODEL_PATH"
echo "================================"
echo "FLARE25 Evaluation"
echo "================================"
echo "Python: $(which python)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "================================"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or edit this script"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path not found: $DATASET_PATH"
    echo "Please set DATASET_PATH environment variable or edit this script"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
if [ -n "$BASELINE_MODEL_PATH" ]; then
    BASELINE_ARG=(--baseline_model_path "$BASELINE_MODEL_PATH")
else
    BASELINE_ARG=()
fi

python evaluation/evaluate_flare25.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    "${BASELINE_ARG[@]}"

echo ""
echo "================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
