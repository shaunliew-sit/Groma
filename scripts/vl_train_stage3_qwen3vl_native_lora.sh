#!/bin/bash
################################################################################
# [GROMA-QWEN V2] Stage 3: LoRA Training with Native Qwen3VL Architecture
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# This script trains the Groma-Qwen V2 model using LoRA (Low-Rank Adaptation):
# - LoRA applied ONLY to Qwen3VL LLM
# - ROI Align + Image-to-Text Bridge remain fully trainable
# - Qwen3VL Vision Encoder frozen
#
# Benefits over full fine-tuning:
# - Aligns output format with training data expectations
# - Prevents catastrophic forgetting of base LLM knowledge
# - Trains only ~0.5% of LLM parameters (50M vs 8B)
# - Enables higher learning rates (2e-5 vs 2e-6)
#
# Usage:
#   bash scripts/vl_train_stage3_qwen3vl_native_lora.sh <STAGE2_CHECKPOINT> <OUTPUT_DIR> [GPU_IDS] [TASK_TYPE]
#
# Examples:
#   # Referring only (output format alignment)
#   bash scripts/vl_train_stage3_qwen3vl_native_lora.sh \
#       checkpoints/groma-qwen-v2-stage2-referring-new \
#       checkpoints/groma-qwen-v2-stage3-lora-referring \
#       "0,7" \
#       referring
#
#   # Grounding only (instruction understanding)
#   bash scripts/vl_train_stage3_qwen3vl_native_lora.sh \
#       checkpoints/groma-qwen-v2-stage2-grounding-json \
#       checkpoints/groma-qwen-v2-stage3-lora-grounding \
#       "0,1,2" \
#       grounding
#
#   # Combined (both tasks)
#   bash scripts/vl_train_stage3_qwen3vl_native_lora.sh \
#       checkpoints/groma-qwen-v2-stage2-combined-json-new \
#       checkpoints/groma-qwen-v2-stage3-lora-combined \
#       "0,1,2" \
#       both
#
# Environment Variables:
#   WANDB_PROJECT     W&B project name (default: groma-qwen-v2-stage3-lora)
#   WANDB_ENTITY      W&B entity/team name
#   NUM_EPOCHS        Number of training epochs (default: 5)
#   BATCH_SIZE        Per-device batch size (default: 8)
#   GRAD_ACCUM        Gradient accumulation steps (default: 2)
#   LR                Learning rate (default: 2e-5)
#   LORA_R            LoRA rank (default: 32)
#   LORA_ALPHA        LoRA alpha scaling (default: 64)
#   LORA_DROPOUT      LoRA dropout (default: 0.1)
#   MAX_STEPS         Max training steps (for quick testing, default: -1 = disabled)
#
# Task Types:
#   referring   - Train on referring task only (action recognition, 153,834 samples)
#   grounding   - Train on grounding task only (JSON format, 128,923 samples)
#   both        - Train on both tasks (282,757 samples)
#
################################################################################

set -e  # Exit on error

# Required arguments
STAGE2_CHECKPOINT=${1:-"checkpoints/groma-qwen-v2-stage2-referring-new"}
OUTPUT_DIR=${2:-"checkpoints/groma-qwen-v2-stage3-lora"}

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
TASK_TYPE=${4:-"referring"}

# Calculate number of GPUs
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# LoRA Configuration (can be overridden via environment variables)
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"}

# Training hyperparameters (can be overridden via environment variables)
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-2e-5}
MAX_STEPS=${MAX_STEPS:--1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SAVE_STEPS=${SAVE_STEPS:-2000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-qwen-v2-stage3-lora"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
RUN_NAME="groma-qwen-v2-stage3-lora-${TASK_TYPE}-$(date +%Y%m%d-%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "[GROMA-QWEN V2] Stage 3 LoRA Training - Native Qwen3VL Architecture"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stage 2 Checkpoint: $STAGE2_CHECKPOINT"
echo "  Output Dir:         $OUTPUT_DIR"
echo "  GPU IDs:            $GPU_IDS"
echo "  Num GPUs:           $ACTUAL_NUM_GPUS"
echo "  Task Type:          $TASK_TYPE"
echo ""
echo "LoRA Configuration:"
echo "  Rank (r):           $LORA_R"
echo "  Alpha:              $LORA_ALPHA"
echo "  Dropout:            $LORA_DROPOUT"
echo "  Target Modules:     $LORA_TARGET_MODULES"
echo ""
echo "Training Parameters:"
echo "  Epochs:             $NUM_EPOCHS"
echo "  Batch Size:         $BATCH_SIZE (per device)"
echo "  Grad Accum:         $GRAD_ACCUM"
echo "  Effective BS:       $((BATCH_SIZE * GRAD_ACCUM * ACTUAL_NUM_GPUS))"
echo "  Learning Rate:      $LR"
echo "  Warmup Ratio:       $WARMUP_RATIO"
echo "  Weight Decay:       $WEIGHT_DECAY"
echo "  Max Steps:          $MAX_STEPS"
echo ""
echo "Architecture (V2 Native + LoRA):"
echo "  Frozen:             Qwen3VL Vision Encoder"
echo "  Fully Trainable:    ROI Align + Image-to-Text Bridge"
echo "  LoRA Adapters:      Qwen3VL LLM (~50M params)"
echo ""
echo "Dataset:"
case $TASK_TYPE in
    referring)
        echo "  Type:           Referring only (action recognition)"
        echo "  Samples:        153,834"
        echo "  Purpose:        Output format alignment"
        ;;
    grounding)
        echo "  Type:           Grounding only (JSON format)"
        echo "  Samples:        128,923"
        echo "  Purpose:        Instruction understanding"
        ;;
    both)
        echo "  Type:           Combined (referring + grounding)"
        echo "  Samples:        282,757"
        echo "  Purpose:        Comprehensive improvement"
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

# Verify Stage 2 checkpoint exists
if [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "ERROR: Stage 2 checkpoint not found at: $STAGE2_CHECKPOINT"
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

echo "✓ Dataset files verified"

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

echo "Starting Stage 3 LoRA training..."
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25011 \
    -m groma.train.train_qwen_v2_lora \
    --model_name_or_path $STAGE2_CHECKPOINT \
    --dataset_config groma/data/configs/vl_train_stage2_qwen3vl_native.py \
    --use_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES" \
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
    --weight_decay $WEIGHT_DECAY \
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
    echo "Stage 3 LoRA Training Complete!"
    echo "========================================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo "Log file:       $LOG_FILE"
    echo "LoRA config:    $OUTPUT_DIR/lora_config.json"
    echo ""
    echo "Note: LoRA weights have been automatically merged into the model."
    echo ""
    echo "Next steps - Evaluation (depending on task type):"
    if [ "$TASK_TYPE" = "referring" ]; then
        echo "  # Referring evaluation (HICO-DET):"
        echo "  bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
        echo ""
        echo "  # Referring evaluation (SWIG-HOI):"
        echo "  bash scripts/run_swig_action_referring_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
    elif [ "$TASK_TYPE" = "grounding" ]; then
        echo "  # Grounding evaluation (HICO-DET):"
        echo "  bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
        echo ""
        echo "  # Grounding evaluation (SWIG-HOI):"
        echo "  bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
    else
        echo "  # Referring evaluation (HICO-DET):"
        echo "  bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
        echo ""
        echo "  # Grounding evaluation (HICO-DET):"
        echo "  bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 $OUTPUT_DIR"
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

