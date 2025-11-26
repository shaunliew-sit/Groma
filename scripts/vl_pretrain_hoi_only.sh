#!/bin/bash
# Stage 2: VL Alignment Pretraining - HICO + SWIG HOI Only
#
# Usage:
#   bash scripts/vl_pretrain_hoi_only.sh <LLM_PATH> <PERCEIVER_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
#
# Arguments:
#   LLM_PATH    : Path to Vicuna-7b-v1.5 checkpoint (required)
#   PERCEIVER   : Path to Groma-det-pretrain from Stage 1 (required)
#   OUTPUT_DIR  : Output directory (required)
#   GPU_IDS     : Comma-separated GPU IDs (e.g., "5,6,7") [default: "0,1,2,3,4,5,6,7"]
#   NUM_GPUS    : Number of GPUs (overrides GPU_IDS if set) [default: 8]
#   RUN_NAME    : WandB run name [default: "vl-pretrain-hoi-<timestamp>"]
#
# Examples:
#   # Use GPUs 5,6,7
#   bash scripts/vl_pretrain_hoi_only.sh \
#       checkpoints/vicuna-7b-v1.5 \
#       checkpoints/groma-det-pretrain-mixed \
#       checkpoints/groma-7b-pretrain-hoi \
#       "5,6,7"
#
#   # Use first 4 GPUs
#   bash scripts/vl_pretrain_hoi_only.sh \
#       checkpoints/vicuna-7b-v1.5 \
#       checkpoints/groma-det-pretrain-mixed \
#       checkpoints/groma-7b-pretrain-hoi \
#       "" \
#       4

# Required arguments
LLM_PATH=$1
PERCEIVER_PATH=$2
OUTPUT_DIR=$3

# Optional arguments
GPU_IDS=${4:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${5:-8}
RUN_NAME=${6:-"vl-pretrain-hoi-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$5" ] && [ "$5" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-vl-alignment-hoi"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 2: VL Alignment - HICO + SWIG HOI Only"
echo "============================================================"
echo "Configuration:"
echo "  LLM Path:      $LLM_PATH"
echo "  Perceiver:     $PERCEIVER_PATH"
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
echo "  Freeze LLM:        True  (Vicuna-7b frozen)"
echo "  Freeze Perceiver:  True  (Detection head frozen)"
echo "  Trainable:         VL Bridge + Region Encoder"
echo "  Epochs:            2"
echo "  Learning Rate:     1e-4"
echo "  Batch Size:        $((2 * ACTUAL_NUM_GPUS * 8)) (2 × $ACTUAL_NUM_GPUS GPUs × 8 accumulation)"
echo ""
echo "Logging:"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Run Name:      $RUN_NAME"
echo "  Log File:      $OUTPUT_DIR/train.log"
echo "============================================================"
echo ""

# Verify checkpoints exist
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ ERROR: LLM checkpoint not found at: $LLM_PATH"
    echo ""
    echo "Please download Vicuna-7b-v1.5:"
    echo "  huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir $LLM_PATH"
    exit 1
fi

if [ ! -d "$PERCEIVER_PATH" ]; then
    echo "❌ ERROR: Perceiver checkpoint not found at: $PERCEIVER_PATH"
    echo ""
    echo "This should be the output from Stage 1 (Detection Pretraining)"
    echo "Expected location: checkpoints/groma-det-pretrain-mixed/pytorch_model.bin"
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
    groma/train/train_mem.py \
    --llm $LLM_PATH \
    --perceiver $PERCEIVER_PATH \
    --dataset_config groma/data/configs/vl_pretrain_hoi_only.py \
    --freeze_perceiver True \
    --freeze_llm True \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 8 \
    --box_score_thres 0.15 \
    2>&1 | tee $OUTPUT_DIR/train.log

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Stage 2 Training completed successfully!"
    echo "============================================================"
    echo "Output:"
    echo "  Checkpoint:    $OUTPUT_DIR/pytorch_model.bin"
    echo "  Log file:      $OUTPUT_DIR/train.log"
    echo "  WandB:         https://wandb.ai/$WANDB_PROJECT/runs/$RUN_NAME"
    echo ""
    echo "Ready for Stage 3: Instruction Finetuning"
    echo "  Use the same HOI datasets for Stage 3"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed! Check: $OUTPUT_DIR/train.log"
    echo "============================================================"
    exit 1
fi
