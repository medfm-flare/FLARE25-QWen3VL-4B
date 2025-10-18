#!/bin/bash
#
# Dry-run script for FLARE25 training
# Runs 5 training steps to validate pipeline without full training
#

set -e  # Exit on error

# ===== Configuration =====
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# DeepSpeed config
deepspeed=./scripts/zero3_offload.json

# Model
llm=Qwen/Qwen3-VL-4B-Instruct

# Training parameters (minimal for dry-run)
lr=1e-5
batch_size=1
grad_accum_steps=2  # Smaller for dry-run

# Use only first dataset with small sample rate for quick test
datasets="flare_neojaundice%10"  # Only 10% of neojaundice

# Training entry
entry_file=qwenvl/train/train_qwen.py

# Output
run_name="qwen3vl_flare25_dryrun"
output_dir=./output/qwen3vl_flare25_dryrun

# Create output directory
mkdir -p ${output_dir}

# ===== Dry-run specific arguments =====
# Override to run only 5 steps
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps 5 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    \
    --max_pixels 50176 \
    --min_pixels 784 \
    \
    --eval_strategy no \
    --save_strategy no \
    \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    \
    --run_name ${run_name}"

# ===== Print Configuration =====
echo "================================================================================"
echo "FLARE25 Qwen3-VL-4B Dry-Run Test"
echo "================================================================================"
echo "Model: ${llm}"
echo "Dataset: ${datasets}"
echo "Output: ${output_dir}"
echo "GPU Count: ${NPROC_PER_NODE}"
echo "Max Steps: 5"
echo "Batch Size: ${batch_size}"
echo "Gradient Accumulation: ${grad_accum_steps}"
echo "Effective Batch Size: $((batch_size * grad_accum_steps))"
echo ""
echo "This dry-run will:"
echo "  1. Load the model and datasets"
echo "  2. Run 5 training steps"
echo "  3. Verify GPU memory usage"
echo "  4. Exit without saving checkpoint"
echo "================================================================================"
echo ""

# Check if output directory already exists from previous run
if [ -d "${output_dir}" ] && [ "$(ls -A ${output_dir})" ]; then
    echo "⚠️  WARNING: Output directory exists from previous run"
    echo "   Cleaning up: ${output_dir}"
    rm -rf ${output_dir}/*
fi

# ===== Launch Dry-run =====
echo "🚀 Starting dry-run..."
echo ""

# Capture start time
start_time=$(date +%s)

# Run training with error handling (using uv for environment management)
if uv run torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}; then

    # Calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "================================================================================"
    echo "✅ DRY-RUN SUCCESSFUL!"
    echo "================================================================================"
    echo "Duration: ${duration} seconds"
    echo ""
    echo "Pipeline validated successfully. Key checks:"
    echo "  ✅ Model loaded correctly"
    echo "  ✅ Data loading working"
    echo "  ✅ Forward/backward pass successful"
    echo "  ✅ GPU memory within limits"
    echo ""
    echo "You can now run full training with:"
    echo "  bash scripts/sft_qwen3_4b_flare.sh"
    echo "================================================================================"

    exit 0
else
    # Calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "================================================================================"
    echo "❌ DRY-RUN FAILED!"
    echo "================================================================================"
    echo "Duration: ${duration} seconds"
    echo ""
    echo "Common issues to check:"
    echo "  1. CUDA out of memory → Reduce batch_size or max_pixels"
    echo "  2. Dataset not found → Check paths in qwenvl/data/__init__.py"
    echo "  3. Model loading error → Verify model_name_or_path"
    echo "  4. DeepSpeed error → Check DeepSpeed installation"
    echo ""
    echo "Check the error logs above for details."
    echo "================================================================================"

    exit 1
fi
