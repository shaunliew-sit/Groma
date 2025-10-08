#!/bin/bash
#
# HICO-DET HOI Instruction Fine-tuning Script
# Fine-tune groma-7b-finetune on HICO-DET HOI grounding instructions
#
# Usage:
#   bash scripts/finetune_hico_hoi.sh [GPU_IDS] [MODEL_PATH] [OUTPUT_DIR] [WANDB_RUN_NAME]
#
# Examples:
#   # Use GPU 0 only
#   bash scripts/finetune_hico_hoi.sh "0" FoundationVision/groma-7b-finetune checkpoints/groma_hico_hoi_finetune "hico-hoi-finetune"
#
#   # Use GPUs 0,2,3
#   bash scripts/finetune_hico_hoi.sh "0,2,3" FoundationVision/groma-7b-finetune checkpoints/groma_hico_hoi_finetune "hico-hoi-3gpu"
#
#   # Use local checkpoint
#   bash scripts/finetune_hico_hoi.sh "0" checkpoints/groma-7b-finetune checkpoints/groma_hico_hoi_finetune "hico-hoi-local"

# Parse arguments
GPU_IDS=${1:-"0"}
MODEL_PATH=${2:-"FoundationVision/groma-7b-finetune"}
OUTPUT_DIR=${3:-"checkpoints/groma_hico_hoi_finetune"}
WANDB_RUN_NAME=${4:-"hico-hoi-finetune-$(date +%Y%m%d)"}

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

# Adjust hyperparameters based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    BATCH_SIZE=2
    GRAD_ACCUM=8
    FSDP_ARGS=""
    GRADIENT_CKPT="False"
elif [ $NUM_GPUS -le 4 ]; then
    BATCH_SIZE=4
    GRAD_ACCUM=4
    FSDP_ARGS="--fsdp full_shard auto_wrap --fsdp_config scripts/fsdp_config.json"
    GRADIENT_CKPT="True"
else
    BATCH_SIZE=16
    GRAD_ACCUM=1
    FSDP_ARGS="--fsdp full_shard auto_wrap --fsdp_config scripts/fsdp_config.json"
    GRADIENT_CKPT="True"
fi

echo "========================================"
echo "HICO-DET HOI Instruction Fine-tuning"
echo "========================================"
echo "Configuration:"
echo "  - Base model: $MODEL_PATH"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Dataset config: groma/data/configs/hoi_finetune.py"
echo "  - WandB project: groma-hoi"
echo "  - WandB run name: $WANDB_RUN_NAME"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  - Training epochs: 1"
echo "  - Gradient checkpointing: $GRADIENT_CKPT"
echo "========================================"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Set WandB project via environment variable
export WANDB_PROJECT=groma-hoi

# Run fine-tuning
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=25002 \
    groma/train/train_mem.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_config groma/data/configs/hoi_finetune.py \
    --freeze_perceiver True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    $FSDP_ARGS \
    --gradient_checkpointing $GRADIENT_CKPT \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $WANDB_RUN_NAME \
    --dataloader_num_workers 4 \
    --box_score_thres 0.15 \
    | tee $OUTPUT_DIR/train.log

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "python -m groma.eval.run_groma_hoi_no_adj \\"
echo "    --model-name $OUTPUT_DIR \\"
echo "    --dataset hico \\"
echo "    --data-root /path/to/hico_20160224_det \\"
echo "    --output-dir results/hico_eval"
