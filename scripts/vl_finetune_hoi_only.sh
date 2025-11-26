#!/bin/bash
# Stage 3: Instruction Finetuning - HICO + SWIG HOI Only
#
# Usage:
#   bash scripts/vl_finetune_hoi_only.sh <PRETRAIN_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
#
# Arguments:
#   PRETRAIN_PATH : Path to Groma-7b-pretrain checkpoint from Stage 2 (required)
#   OUTPUT_DIR    : Output directory (required)
#   GPU_IDS       : Comma-separated GPU IDs (e.g., "5,6,7") [default: "0,1,2,3,4,5,6,7"]
#   NUM_GPUS      : Number of GPUs (overrides GPU_IDS if set) [default: 8]
#   RUN_NAME      : WandB run name [default: "vl-finetune-hoi-<timestamp>"]
#
# Examples:
#   # Use GPUs 5,6,7
#   bash scripts/vl_finetune_hoi_only.sh \
#       groma-7b-finetune \
#       checkpoints/groma-7b-hoi-ground-refer-fine-tuned \
#       "5,6,7"
#   bash scripts/vl_finetune_hoi_only.sh \
#       groma-7b-finetune \
#       checkpoints/groma-7b-hoi-ground-refer-fine-tuned \
#       "1,4"
#
#   # Use first 4 GPUs
#   bash scripts/vl_finetune_hoi_only.sh \
#       checkpoints/groma-7b-pretrain-hoi \
#       checkpoints/groma-7b-finetune-hoi \
#       "" \
#       4

# Required arguments
PRETRAIN_PATH=$1
OUTPUT_DIR=$2

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${4:-8}
RUN_NAME=${5:-"vl-finetune-hoi-ground-refer-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$4" ] && [ "$4" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-instruction-ground-refer-finetune-hoi"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 3: Instruction Finetuning - HICO + SWIG HOI Only"
echo "============================================================"
echo "Configuration:"
echo "  Pretrain Path: $PRETRAIN_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo ""
echo "Datasets:"
echo "  HICO-DET:      112,899 instruction samples"
echo "  SWIG-HOI:      163,803 instruction samples"
echo "  Total:         276,702 samples"
echo ""
echo "Training:"
echo "  Freeze LLM:        False  (LLM now trainable!)"
echo "  Freeze Perceiver:  True   (Detection head frozen)"
echo "  Trainable:         LLM + VL Bridge + Region Encoder"
echo "  Epochs:            1"
echo "  Learning Rate:     2e-5 (lower than Stage 2)"
echo "  Batch Size:        $((16 * ACTUAL_NUM_GPUS)) (16 × $ACTUAL_NUM_GPUS GPUs)"
echo "  FSDP:              Enabled (memory efficient)"
echo ""
echo "Logging:"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Run Name:      $RUN_NAME"
echo "  Log File:      $OUTPUT_DIR/train.log"
echo "============================================================"
echo ""

# Verify checkpoint exists
if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "❌ ERROR: Pretrain checkpoint not found at: $PRETRAIN_PATH"
    echo ""
    echo "This should be the output from Stage 2 (VL Alignment)"
    echo "Expected location: checkpoints/groma-7b-pretrain-hoi/pytorch_model.bin"
    exit 1
fi

# Set WandB environment
export WANDB_PROJECT=$WANDB_PROJECT
if [ ! -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=$WANDB_ENTITY
fi
#groma/data/configs/vl_finetune_hoi_only.py
# Run training
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25003 \
    groma/train/train_mem.py \
    --model_name_or_path $PRETRAIN_PATH \
    --dataset_config groma/data/configs/hico_swig_mixed_finetune.py \
    --freeze_perceiver True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config scripts/fsdp_config.json \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 4 \
    --box_score_thres 0.15 \
    2>&1 | tee $OUTPUT_DIR/train.log

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Stage 3 Training completed successfully!"
    echo "============================================================"
    echo "Output:"
    echo "  Checkpoint:    $OUTPUT_DIR/pytorch_model.bin"
    echo "  Log file:      $OUTPUT_DIR/train.log"
    echo "  WandB:         https://wandb.ai/$WANDB_PROJECT/runs/$RUN_NAME"
    echo ""
    echo "🎉 Full Training Pipeline Complete!"
    echo "============================================================"
    echo "Your HOI-specialized Groma model is ready for evaluation!"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate on HICO-DET test set"
    echo "  2. Evaluate on SWIG-HOI test set"
    echo "  3. Compare with official Groma baseline (if trained both)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed! Check: $OUTPUT_DIR/train.log"
    echo "============================================================"
    exit 1
fi
