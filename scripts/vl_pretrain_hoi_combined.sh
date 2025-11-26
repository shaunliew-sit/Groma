#!/bin/bash
# Stage 2: VL Alignment Pretraining - HICO + SWIG Combined
#
# Usage:
#   bash scripts/vl_pretrain_hoi_combined.sh <LLM_PATH> <PERCEIVER_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
"""
bash scripts/vl_pretrain_hoi_combined.sh \
    checkpoints/Qwen3-VL-8B-Instruct \
    checkpoints/dinov2-large \
    checkpoints/groma-qwen-stage2-hoi-qwen3vl-24-nov-fixed \
    "0,6,7"
"""
# Required arguments
LLM_PATH=$1
PERCEIVER_PATH=$2
OUTPUT_DIR=$3

# Optional arguments
GPU_IDS=${4:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${5:-8}
RUN_NAME=${6:-"vl-pretrain-hoi-combined-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$5" ] && [ "$5" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-vl-alignment-hoi-combined"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 2: VL Alignment - HICO + SWIG Combined"
echo "============================================================"
echo "Configuration:"
echo "  LLM Path:      $LLM_PATH"
echo "  Perceiver:     $PERCEIVER_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo ""
echo "Training:"
echo "  Freeze LLM:            True"
echo "  Freeze Perceiver:      True"
echo "  Trainable:             VL Bridge + Region Encoder"
echo "  Gradient Checkpointing: True"
echo "  Dataset Config:        groma/data/configs/vl_pretrain_hoi_combined.py"
echo "============================================================"

# Verify checkpoints exist
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ ERROR: LLM checkpoint not found at: $LLM_PATH"
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
    --master_port=25003 \
    groma/train/train_qwen_mem.py \
    --llm $LLM_PATH \
    --perceiver $PERCEIVER_PATH \
    --dataset_config groma/data/configs/vl_pretrain_hoi_combined.py \
    --freeze_perceiver True \
    --freeze_llm True \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 8 \
    --box_score_thres 0.15 \
    2>&1 | tee $OUTPUT_DIR/train.log
