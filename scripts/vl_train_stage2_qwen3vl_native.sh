#!/bin/bash
################################################################################
# [GROMA-QWEN V2] Stage 2: Training with Native Qwen3VL Architecture
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# This script trains the Groma-Qwen V2 model using only native Qwen3VL components:
# - Reuses Qwen3VL Vision Encoder (no DINOv2)
# - Referring: Uses native bbox tokens (<|box_start|>, <|box_end|>)
# - Grounding: Uses JSON format ({"bbox_2d": [...], "label": "..."}) aligned with Qwen3-VL
# - Trains only ROI Align + Image-to-Text Bridge
#
# Usage:
#   bash scripts/vl_train_stage2_qwen3vl_native.sh <LLM_PATH> <OUTPUT_DIR> [GPU_IDS] [TASK_TYPE]
#
# Examples:
#   # Grounding training with JSON format
#   bash scripts/vl_train_stage2_qwen3vl_native.sh \
#       checkpoints/Qwen3-VL-8B-Instruct \
#       checkpoints/groma-qwen-v2-stage2-grounding-json \
#       "4,5,6" \
#       grounding
#
#   # Combined training (referring + grounding with JSON)
#   bash scripts/vl_train_stage2_qwen3vl_native.sh \
#       checkpoints/Qwen3-VL-8B-Instruct \
#       checkpoints/groma-qwen-v2-stage2-combined \
#       "1,4,5" \
#       both
#
#   # Single GPU test
#   bash scripts/vl_train_stage2_qwen3vl_native.sh \
#       checkpoints/Qwen3-VL-8B-Instruct \
#       checkpoints/groma-qwen-v2-test \
#       "0" \
#       referring
#
# Environment Variables:
#   WANDB_PROJECT     W&B project name (default: groma-qwen-v2-training)
#   WANDB_ENTITY      W&B entity/team name
#   NUM_EPOCHS        Number of training epochs (default: 4)
#   BATCH_SIZE        Per-device batch size (default: 4)
#   GRAD_ACCUM        Gradient accumulation steps (default: 4)
#   LR                Learning rate (default: 2e-4)
#   MAX_STEPS         Max training steps (for quick testing, default: -1 = disabled)
#
# Task Types:
#   referring   - Train on referring task only (153,834 samples)
#   grounding   - Train on grounding task only with JSON format (128,923 samples)
#   both        - Train on both tasks (282,757 samples)
#
################################################################################

set -e  # Exit on error

# Required arguments
LLM_PATH=${1:-"checkpoints/Qwen3-VL-8B-Instruct"}
OUTPUT_DIR=${2:-"checkpoints/groma-qwen-v2-stage2"}

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
TASK_TYPE=${4:-"referring"}

# Calculate number of GPUs
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# Training hyperparameters (can be overridden via environment variables)
NUM_EPOCHS=${NUM_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
MAX_STEPS=${MAX_STEPS:--1}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-2500}

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-qwen-v2-training"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
RUN_NAME="groma-qwen-v2-${TASK_TYPE}-$(date +%Y%m%d-%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "[GROMA-QWEN V2] Stage 2 Training - Native Qwen3VL Architecture"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  LLM Path:       $LLM_PATH"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  GPU IDs:        $GPU_IDS"
echo "  Num GPUs:       $ACTUAL_NUM_GPUS"
echo "  Task Type:      $TASK_TYPE"
echo ""
echo "Training Parameters:"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Batch Size:     $BATCH_SIZE (per device)"
echo "  Grad Accum:     $GRAD_ACCUM"
echo "  Effective BS:   $((BATCH_SIZE * GRAD_ACCUM * ACTUAL_NUM_GPUS))"
echo "  Learning Rate:  $LR"
echo "  Warmup Ratio:   $WARMUP_RATIO"
echo "  Max Steps:      $MAX_STEPS"
echo ""
echo "Architecture (V2 Native):"
echo "  Freeze LLM:            True"
echo "  Freeze Vision:         True (features extracted)"
echo "  Trainable:             ROI Align + Image-to-Text Bridge"
echo "  Referring Format:      Native bbox tokens (<|box_start|>, <|box_end|>)"
echo "  Grounding Format:      JSON ({\"bbox_2d\": [...], \"label\": \"...\"})"
echo "  Vision Encoder:        Qwen3VL (1152-dim)"
echo ""
echo "Dataset:"
case $TASK_TYPE in
    referring)
        echo "  Type:           Referring only (native tokens)"
        echo "  Samples:        153,834"
        ;;
    grounding)
        echo "  Type:           Grounding only (dedicated JSON files)"
        echo "  Files:          hico_grounding_only_json.json + swig_grounding_only_json.json"
        echo "  Samples:        128,923 (69,404 + 59,519)"
        echo "  Format:         [{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"person/object\"}]"
        ;;
    both)
        echo "  Type:           Combined (referring + grounding with JSON)"
        echo "  Samples:        282,757"
        ;;
    *)
        echo "ERROR: Invalid TASK_TYPE. Must be 'referring', 'grounding', or 'both'"
        exit 1
        ;;
esac
echo "  Config:         groma/data/configs/vl_train_stage2_qwen3vl_native.py"
echo ""
echo "WandB:"
echo "  Project:        $WANDB_PROJECT"
echo "  Run Name:       $RUN_NAME"
echo ""
echo "Log File:         $LOG_FILE"
echo "========================================================================"
echo ""

# Verify LLM checkpoint exists
if [ ! -d "$LLM_PATH" ]; then
    echo "ERROR: LLM checkpoint not found at: $LLM_PATH"
    exit 1
fi

# Verify dataset files exist
if [ "$TASK_TYPE" = "referring" ]; then
    if [ ! -f "groma_data/benchmarks/hico_referring_qwen3vl_native.json" ]; then
        echo "ERROR: Dataset not found. Run regeneration script first:"
        echo "  python scripts/regenerate_datasets_qwen3vl_native.py --task referring"
        exit 1
    fi
elif [ "$TASK_TYPE" = "grounding" ]; then
    # For grounding-only, use dedicated grounding files
    if [ ! -f "groma_data/benchmarks/hico_grounding_only_json.json" ]; then
        echo "ERROR: Grounding-only dataset not found. Run extraction script first:"
        echo "  python scripts/extract_grounding_only_datasets.py"
        exit 1
    fi
else
    # For both, use combined JSON format datasets
    if [ ! -f "groma_data/benchmarks/hico_combined_qwen3vl_json.json" ]; then
        echo "ERROR: Combined JSON dataset not found. Run conversion script first:"
        echo "  python scripts/convert_grounding_to_json_format.py --convert-all"
        exit 1
    fi
fi

# Set environment variables
export TASK_TYPE=$TASK_TYPE
export WANDB_PROJECT=$WANDB_PROJECT
if [ ! -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=$WANDB_ENTITY
fi

# Build max_steps argument
MAX_STEPS_ARG=""
if [ "$MAX_STEPS" != "-1" ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi

echo "Starting training..."
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25010 \
    -m groma.train.train_qwen \
    --llm $LLM_PATH \
    --dataset_config groma/data/configs/vl_train_stage2_qwen3vl_native.py \
    --freeze_llm True \
    --freeze_vl_bridge False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 8 \
    $MAX_STEPS_ARG \
    2>&1 | tee $LOG_FILE

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Training Complete!"
    echo "========================================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo "Log file:       $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Test inference:"
    echo "     python scripts/test_groma_qwen_referring.py \\"
    echo "         --model-name $OUTPUT_DIR \\"
    echo "         --base-model-name $LLM_PATH \\"
    echo "         --image-file <IMAGE_PATH> \\"
    echo "         --subject-bbox \"x1,y1,x2,y2\" \\"
    echo "         --object-bbox \"x1,y1,x2,y2\""
    echo ""
    echo "  2. Run evaluation (depending on task type):"
    if [ "$TASK_TYPE" = "referring" ]; then
        echo "     bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
    elif [ "$TASK_TYPE" = "grounding" ]; then
        echo "     bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
        echo "     # Note: Uses JSON format for grounding evaluation"
    else
        echo "     # Referring:"
        echo "     bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
        echo "     # Grounding (JSON format):"
        echo "     bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
    fi
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Training failed!"
    echo "========================================================================"
    echo "Check the log file for details:"
    echo "  $LOG_FILE"
    echo "========================================================================"
    exit 1
fi

