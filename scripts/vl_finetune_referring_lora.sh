#!/bin/bash
# [GROMA-QWEN] Stage 3: LoRA Fine-tuning - Referring Only
# Part of: Groma Qwen3VL Referring Task Implementation
#
# This script trains using LoRA (Low-Rank Adaptation) instead of full fine-tuning.
# LoRA is applied ONLY to the Qwen3VL LLM, while Groma components remain fully trainable.
#
# Benefits over full fine-tuning:
# - Prevents catastrophic forgetting of base LLM knowledge
# - Trains only ~0.5% of LLM parameters (50M vs 8B)
# - Enables higher learning rates (1e-4 vs 2e-6)
# - Reduces memory usage and training time
#
# Usage:
#   bash scripts/vl_finetune_referring_lora.sh <PRETRAIN_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
#
# Example:
"""
bash scripts/vl_finetune_referring_lora.sh \
    checkpoints/groma-qwen-stage2-hoi-qwen3vl-24-nov-fixed \
    checkpoints/groma-qwen-stage3-lora-referring \
    "1,6,7"
"""

# Required arguments
PRETRAIN_PATH=$1
OUTPUT_DIR=$2

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${4:-8}
RUN_NAME=${5:-"vl-finetune-lora-referring-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$4" ] && [ "$4" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-lora-finetune-referring"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

# LoRA Configuration (can be overridden via environment variables)
# Note: Smaller rank (32) with proportional alpha provides better regularization
# and prevents overfitting on the HICO/SWIG referring task
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"}

# Training hyperparameters (can be overridden via environment variables)
# Lower learning rate (2e-5) is more stable for LoRA than 1e-4
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 3: LoRA Fine-tuning - Referring Only"
echo "============================================================"
echo "Configuration:"
echo "  Pretrain Path: $PRETRAIN_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo ""
echo "LoRA Configuration:"
echo "  Rank (r):       $LORA_R"
echo "  Alpha:          $LORA_ALPHA"
echo "  Dropout:        $LORA_DROPOUT"
echo "  Target Modules: $LORA_TARGET_MODULES"
echo ""
echo "Training Hyperparameters:"
echo "  Learning Rate:     $LEARNING_RATE"
echo "  Num Epochs:        $NUM_EPOCHS"
echo "  Batch Size:        $BATCH_SIZE"
echo "  Grad Accumulation: $GRAD_ACCUM"
echo "  Effective Batch:   $((BATCH_SIZE * GRAD_ACCUM * ACTUAL_NUM_GPUS))"
echo ""
echo "Model Configuration:"
echo "  LoRA on LLM:       True"
echo "  Freeze Perceiver:  True (DINOv2 frozen)"
echo "  Fully Trainable:   region_encoder, img_txt_bridge, new_input_embs"
echo "  Dataset Config:    groma/data/configs/vl_finetune_referring_only_qwen.py"
echo ""
echo "Dataset:"
echo "  HICO-DET Referring: 93,041 samples"
echo "  SWIG-HOI Referring: 60,793 samples"
echo "  Total:              153,834 samples"
echo "============================================================"

# Verify checkpoint exists
if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "❌ ERROR: Pretrain checkpoint not found at: $PRETRAIN_PATH"
    exit 1
fi

# Verify dataset files exist
HICO_DATASET="groma_data/benchmarks/hico_referring_only_qwen3vl.json"
SWIG_DATASET="groma_data/benchmarks/swig_referring_only_qwen3vl.json"

if [ ! -f "$HICO_DATASET" ]; then
    echo "❌ ERROR: HICO referring dataset not found at: $HICO_DATASET"
    echo "   Run: python scripts/regenerate_referring_only_qwen3vl.py"
    exit 1
fi

if [ ! -f "$SWIG_DATASET" ]; then
    echo "❌ ERROR: SWIG referring dataset not found at: $SWIG_DATASET"
    echo "   Run: python scripts/regenerate_referring_only_qwen3vl.py"
    exit 1
fi

echo "✓ Dataset files found"

# Set WandB environment
export WANDB_PROJECT=$WANDB_PROJECT
if [ ! -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=$WANDB_ENTITY
fi

# Run training with LoRA
# Note: Using DDP instead of FSDP for better LoRA compatibility
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25006 \
    groma/train/train_qwen_lora_mem.py \
    --model_name_or_path $PRETRAIN_PATH \
    --dataset_config groma/data/configs/vl_finetune_referring_only_qwen.py \
    --use_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --freeze_perceiver True \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing False \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 8 \
    --max_grad_norm 1.0 \
    --box_score_thres 0.15 \
    2>&1 | tee $OUTPUT_DIR/train.log

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ LoRA Training completed successfully!"
    echo "============================================================"
    echo "Output:"
    echo "  Checkpoint:    $OUTPUT_DIR"
    echo "  Log file:      $OUTPUT_DIR/train.log"
    echo "  LoRA config:   $OUTPUT_DIR/lora_config.json"
    echo ""
    echo "Note: LoRA weights have been automatically merged into the model."
    echo ""
    echo "Next step - Evaluate on HICO-DET test set:"
    echo "  python groma/eval/eval_hico_action_referring_groma_qwen.py \\"
    echo "      --model-name $OUTPUT_DIR \\"
    echo "      --img-prefix ../data/hico_20160224_det/images/test2015 \\"
    echo "      --ann-file groma_data/benchmarks/hico_action_referring_test.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed! Check the log file for errors:"
    echo "   $OUTPUT_DIR/train.log"
    echo "============================================================"
    exit 1
fi

