#!/bin/bash
# Mixed HICO + SWIG Detection Pretraining Script
# Trains on combined HICO-DET (37,633 images) + SWIG-HOI (54,601 images) = 92,234 images
#
# Usage:
#   bash scripts/det_pretrain_mixed.sh <DINO_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
#
# Arguments:
#   DINO_PATH   : Path to DINOv2 checkpoint (required)
#   OUTPUT_DIR  : Path to output directory (required)
#   GPU_IDS     : Comma-separated GPU IDs (e.g., "0,1,2,3" or "5,6,7") [default: "0,1,2,3,4,5,6,7"]
#   NUM_GPUS    : Number of GPUs to use (overrides GPU_IDS if set) [default: 8]
#   RUN_NAME    : WandB run name [default: "hico-swig-det-<timestamp>"]
#
# Examples:
#   # Use all 8 GPUs (0-7)
#   bash scripts/det_pretrain_mixed.sh checkpoints/dinov2-large checkpoints/groma-det-pretrain-mixed
#
#   # Use specific GPUs (5,6,7)
#   bash scripts/det_pretrain_mixed.sh checkpoints/dinov2-large checkpoints/groma-det-pretrain-mixed "5,6,7"
#
#   # Use 4 GPUs starting from GPU 0
#   bash scripts/det_pretrain_mixed.sh checkpoints/dinov2-large checkpoints/groma-det-pretrain-mixed "" 4
#
#   # Use GPUs 5,6,7 with custom run name
#   bash scripts/det_pretrain_mixed.sh checkpoints/dinov2-large checkpoints/groma-det-pretrain-mixed "5,6,7" "" "my-experiment"

# Required arguments
DINO_PATH=$1
OUTPUT_DIR=$2

# Optional arguments with defaults
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${4:-8}
RUN_NAME=${5:-"hico-swig-detection-$(date +%Y%m%d-%H%M%S)"}

# If NUM_GPUS is explicitly provided, override GPU_IDS
if [ ! -z "$4" ] && [ "$4" != "" ]; then
    # Generate GPU_IDS based on NUM_GPUS (e.g., NUM_GPUS=4 -> "0,1,2,3")
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi

# Count actual number of GPUs from GPU_IDS
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-hoi"}
WANDB_ENTITY=${WANDB_ENTITY:-""}  # Leave empty to use default entity

# Create output directory
mkdir -p $OUTPUT_DIR

# Print configuration
echo "============================================================"
echo "Mixed Detection Pretraining: HICO-DET + SWIG-HOI"
echo "============================================================"
echo "Configuration:"
echo "  DINOv2 Path:   $DINO_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo "  Master Port:   25003"
echo ""
echo "Dataset:"
echo "  HICO-DET:      37,633 images, 152,785 boxes"
echo "  SWIG-HOI:      54,601 images, 134,743 boxes"
echo "  Combined:      92,234 images, 287,528 boxes"
echo ""
echo "Logging:"
echo "  WandB Project: $WANDB_PROJECT"
echo "  WandB Entity:  ${WANDB_ENTITY:-[default]}"
echo "  Run Name:      $RUN_NAME"
echo "  Log File:      $OUTPUT_DIR/train.log"
echo "============================================================"
echo ""

# Verify DINOv2 checkpoint exists
if [ ! -d "$DINO_PATH" ]; then
    echo "❌ ERROR: DINOv2 checkpoint not found at: $DINO_PATH"
    exit 1
fi

# Set WandB environment variables
export WANDB_PROJECT=$WANDB_PROJECT
if [ ! -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=$WANDB_ENTITY
fi

# Run training with selected GPUs
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nnodes=1 \
    --nproc_per_node=$ACTUAL_NUM_GPUS \
    --master_port=25003 \
    groma/train/train_det.py \
    --vis_encoder $DINO_PATH \
    --dataset_config groma/data/configs/hico_swig_det_pretrain.py \
    --bf16 True \
    --tf32 True \
    --num_classes 1 \
    --num_queries 300 \
    --two_stage True \
    --with_box_refine True \
    --ddetr_hidden_dim 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_feature_levels 1 \
    --freeze_vis_encoder True \
    --num_train_epochs 12 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 100 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --dataloader_num_workers 8 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/train.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Output:"
    echo "  Checkpoint:    $OUTPUT_DIR/pytorch_model.bin"
    echo "  Log file:      $OUTPUT_DIR/train.log"
    echo "  WandB run:     https://wandb.ai/$WANDB_PROJECT/runs/$RUN_NAME"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed! Check log file:"
    echo "  $OUTPUT_DIR/train.log"
    echo "============================================================"
    exit 1
fi
