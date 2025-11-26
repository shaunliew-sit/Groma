#!/bin/bash
# Stage 3: Instruction Finetuning - HICO + SWIG Combined (v3)
#
# Usage:
#   bash scripts/vl_finetune_hoi_combined_v3.sh <PRETRAIN_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
"""
bash scripts/vl_finetune_hoi_combined_v3.sh \
    checkpoints/groma-qwen-stage2-hoi-qwen3vl-24-nov-fixed \
    checkpoints/groma-qwen-stage3-hoi-qwen3vl-24-nov-fixed \
    "0,6,7"
"""
# Required arguments
PRETRAIN_PATH=$1
OUTPUT_DIR=$2

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${4:-8}
RUN_NAME=${5:-"vl-finetune-hoi-combined-v3-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$4" ] && [ "$4" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-instruction-finetune-hoi-combined"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 3: Instruction Finetuning - HICO + SWIG Combined (v3)"
echo "============================================================"
echo "Configuration:"
echo "  Pretrain Path: $PRETRAIN_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo ""
echo "Training:"
echo "  Freeze LLM:        False"
echo "  Freeze Perceiver:  True"
echo "  Trainable:         LLM + VL Bridge + Region Encoder"
echo "  Dataset Config:    groma/data/configs/vl_finetune_hoi_combined_qwen.py"
echo "============================================================"

# Verify checkpoint exists
if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "❌ ERROR: Pretrain checkpoint not found at: $PRETRAIN_PATH"
    exit 1
fi

# Set WandB environment
export WANDB_PROJECT=$WANDB_PROJECT
if [ ! -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=$WANDB_ENTITY
fi

# Run training
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25004 \
    groma/train/train_qwen_mem.py \
    --model_name_or_path $PRETRAIN_PATH \
    --dataset_config groma/data/configs/vl_finetune_hoi_combined_qwen.py \
    --freeze_perceiver True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config scripts/fsdp_config.json \
    --gradient_checkpointing False \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 8 \
    --max_grad_norm 0.5 \
    --box_score_thres 0.15 \
    2>&1 | tee $OUTPUT_DIR/train.log
