#!/bin/bash
#
# HICO-DET Detection Fine-tuning Training Script
#
# Usage:
#   bash scripts/train_hico_det.sh [GPU_IDS] [DINOV2_PATH] [OUTPUT_DIR] [WANDB_RUN_NAME]
#
# Examples:
#   # Use GPUs 0,2,3
#   bash scripts/train_hico_det.sh "0,2,3" checkpoints/dinov2-large checkpoints/groma_hico_det "hico-det-3gpu"
#
#   # Use GPU 2 only
#   bash scripts/train_hico_det.sh "2" checkpoints/dinov2-large checkpoints/groma_hico_det "hico-det-1gpu"
#
#   # Use all available GPUs
#   bash scripts/train_hico_det.sh "all" checkpoints/dinov2-large checkpoints/groma_hico_det "hico-det-allgpu"

# Parse arguments
GPU_IDS=${1:-"0"}
DINO_PATH=${2:-"checkpoints/dinov2-large"}
OUTPUT_DIR=${3:-"checkpoints/groma_hico_det"}
WANDB_RUN_NAME=${4:-"hico-det-$(date +%Y%m%d)"}

# Determine number of GPUs
if [ "$GPU_IDS" = "all" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    unset CUDA_VISIBLE_DEVICES
    echo "Using all $NUM_GPUS GPUs"
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "Using GPUs: $GPU_IDS (total: $NUM_GPUS)"
fi

# Adjust batch size based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    BATCH_SIZE=4
    GRAD_ACCUM=2
else
    BATCH_SIZE=8
    GRAD_ACCUM=1
fi

echo "Configuration:"
echo "  - DINOv2 checkpoint: $DINO_PATH"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - WandB project: groma-hoi"
echo "  - WandB run name: $WANDB_RUN_NAME"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=25001 \
    groma/train/train_det.py \
    --vis_encoder $DINO_PATH \
    --dataset_config groma/data/configs/hico_det_pretrain.py \
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
    --logging_steps 50 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --dataloader_num_workers 4 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --report_to wandb \
    --wandb_project groma-hoi \
    --run_name $WANDB_RUN_NAME \
    --output_dir $OUTPUT_DIR \
    | tee $OUTPUT_DIR/train.log
