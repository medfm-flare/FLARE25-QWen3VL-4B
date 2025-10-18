#!/bin/bash
#
# FLARE25 Fine-tuning Script for Qwen3-VL-4B
# Optimized for single GPU (24GB VRAM)
#
# Training 5 starter datasets:
# - neojaundice (Clinical, multi-image)
# - retino (Retinography)
# - BUSI-det (Ultrasound, detection)
# - boneresorption (X-ray, regression)
# - bone_marrow (Microscopy)
#

# ===== Distributed Training Configuration =====
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}  # Single GPU

# ===== DeepSpeed Configuration =====
# Use ZeRO-3 with offloading for memory efficiency
deepspeed=./scripts/zero3_offload.json

# ===== Model Configuration =====
llm=Qwen/Qwen3-VL-4B-Instruct

# ===== Training Hyperparameters =====
# Optimized for single 24GB GPU
lr=1e-5                      # Learning rate
batch_size=1                 # Small batch size for 4B model on 24GB
grad_accum_steps=16          # Effective batch size = 1 * 16 = 16

# ===== Dataset Configuration =====
# Use all 5 starter datasets with 100% sampling
# Format: dataset_name%sampling_rate
datasets="flare_neojaundice%100,flare_retino%100,flare_busi_det%100,flare_boneresorption%100,flare_bone_marrow%100"
eval_datasets="flare_neojaundice_val%100,flare_retino_val%100,flare_busi_det_val%100,flare_boneresorption_val%100,flare_bone_marrow_val%100"

# ===== Training Entry Point =====
entry_file=qwenvl/train/train_qwen.py

# ===== Output Configuration =====
run_name="qwen3vl_flare25_starter"
output_dir=./output/qwen3vl_flare25_starter

# Create output directory
mkdir -p ${output_dir}

# ===== Training Arguments =====
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten True \
    \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    \
    --max_pixels 50176 \
    --min_pixels 784 \
    \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
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
    --dataloader_num_workers 4 \
    \
    --run_name ${run_name} \
    --report_to wandb"

# ===== Print Configuration =====
echo "================================================================================"
echo "FLARE25 Qwen3-VL-4B Training Configuration"
echo "================================================================================"
echo "Model: ${llm}"
echo "Datasets: ${datasets}"
echo "Output: ${output_dir}"
echo "GPU Count: ${NPROC_PER_NODE}"
echo "Batch Size: ${batch_size}"
echo "Gradient Accumulation: ${grad_accum_steps}"
echo "Effective Batch Size: $((batch_size * grad_accum_steps))"
echo "Learning Rate: ${lr}"
echo "Epochs: 1.0"
echo "================================================================================"
echo ""

# ===== Launch Training =====
# Using uv for environment management
uv run torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "Model saved to: ${output_dir}"
echo "================================================================================"
