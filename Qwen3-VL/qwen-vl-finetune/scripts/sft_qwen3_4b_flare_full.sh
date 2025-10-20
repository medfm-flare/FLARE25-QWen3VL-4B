#!/bin/bash
#
# FLARE25 FULL Fine-tuning Script for Qwen3-VL-4B
# Training on ALL 19 FLARE25 datasets
#
# Total training samples: ~52,655
# Total validation samples: ~8,450
#

# ===== Distributed Training Configuration =====
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}  # Use 2 GPUs (2x RTX A6000)

# ===== DeepSpeed Configuration =====
deepspeed=./scripts/zero3_offload.json

# ===== Model Configuration =====
llm=Qwen/Qwen3-VL-4B-Instruct

# ===== Training Hyperparameters =====
# Optimized for 2x 48GB GPUs
lr=1e-5                      # Learning rate
batch_size=2                 # Batch size per GPU
grad_accum_steps=8           # Effective batch size = 2 GPUs * 2 * 8 = 32

# ===== Dataset Configuration =====
# ALL 19 FLARE25 datasets with 100% sampling
datasets="flare_neojaundice%100,flare_retino%100,flare_busi_det%100,flare_boneresorption%100,flare_bone_marrow%100,flare_fundus%100,flare_bus_uclm_det%100,flare_busi%100,flare_bus_uclm%100,flare_iugc%100,flare_dental%100,flare_periapical%100,flare_iu_xray%100,flare_chestdr%100,flare_chromosome%100,flare_neurips22cell%100,flare_endo%100,flare_bcn20000%100,flare_cmmd%100"

eval_datasets="flare_neojaundice_val%100,flare_retino_val%100,flare_busi_det_val%100,flare_boneresorption_val%100,flare_bone_marrow_val%100,flare_fundus_val%100,flare_bus_uclm_det_val%100,flare_busi_val%100,flare_bus_uclm_val%100,flare_iugc_val%100,flare_dental_val%100,flare_periapical_val%100,flare_iu_xray_val%100,flare_chestdr_val%100,flare_chromosome_val%100,flare_neurips22cell_val%100,flare_endo_val%100,flare_bcn20000_val%100,flare_cmmd_val%100"

# ===== Training Entry Point =====
entry_file=qwenvl/train/train_qwen.py

# ===== Output Configuration =====
run_name="qwen3vl_flare25"
output_dir=./output/qwen3vl_flare25

# Create output directory
mkdir -p ${output_dir}

# ===== WandB Configuration =====
export WANDB_PROJECT="flare25_qwen3vl"  # New project name
export WANDB_NAME="${run_name}"

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
    --num_train_epochs 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_accumulation_steps 4 \
    \
    --max_pixels 50176 \
    --min_pixels 784 \
    \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
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
echo "FLARE25 Qwen3-VL-4B FULL Training Configuration"
echo "================================================================================"
echo "Model: ${llm}"
echo "Training Datasets: 19 FLARE25 datasets (~52,655 samples)"
echo "Validation Datasets: 19 FLARE25 datasets (~8,450 samples)"
echo "Output: ${output_dir}"
echo "GPU Count: ${NPROC_PER_NODE}"
echo "Batch Size per GPU: ${batch_size}"
echo "Gradient Accumulation: ${grad_accum_steps}"
echo "Effective Batch Size: $((NPROC_PER_NODE * batch_size * grad_accum_steps))"
echo "Learning Rate: ${lr}"
echo "Epochs: 1.0"
echo "WandB Project: flare25_qwen3vl"
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
