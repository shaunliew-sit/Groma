#!/bin/bash
#
# HICO-DET HOI Instruction Fine-tuning Script (Memory Optimized v2)
# Fine-tune groma-7b-finetune on HICO-DET HOI grounding instructions with memory optimizations
#
# Usage:
#   bash scripts/finetune_hico_hoi_memory_optimized_v2.sh [GPU_IDS] [MODEL_PATH] [OUTPUT_DIR] [WANDB_RUN_NAME]

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

# Memory-optimized hyperparameters for single GPU
if [ $NUM_GPUS -eq 1 ]; then
    BATCH_SIZE=1
    GRAD_ACCUM=32  # Reduced from 64 for faster iteration while keeping effective batch size reasonable
    GRADIENT_CKPT="True"
    DATALOADER_WORKERS=2  # Increased for better GPU utilization
    SAVE_STEPS=500
    MODEL_MAX_LENGTH=1536
    LEARNING_RATE=2e-5  # Higher LR since perceiver is now trainable
else
    BATCH_SIZE=2
    GRAD_ACCUM=16
    GRADIENT_CKPT="True"
    DATALOADER_WORKERS=2
    SAVE_STEPS=500
    MODEL_MAX_LENGTH=2048
    LEARNING_RATE=2e-5
fi

# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS=2  # Increased from 1 for better CPU utilization

echo "========================================"
echo "HICO-DET HOI Instruction Fine-tuning (Memory Optimized v2 - FIXED)"
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
echo "  - Freeze perceiver: False (FIXED - allows HOI grounding learning)"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Model max length: $MODEL_MAX_LENGTH"
echo "  - Gradient checkpointing: $GRADIENT_CKPT"
echo "  - DataLoader workers: $DATALOADER_WORKERS"
echo "  - Save steps: $SAVE_STEPS"
echo "  - PyTorch CUDA allocation config: $PYTORCH_CUDA_ALLOC_CONF"
echo "========================================"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Set WandB project via environment variable
export WANDB_PROJECT=groma-hoi

# Clear GPU cache before starting
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Run fine-tuning with memory optimizations
if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training without torchrun
    python groma/train/train_mem.py \
        --model_name_or_path $MODEL_PATH \
        --dataset_config groma/data/configs/hoi_finetune.py \
        --freeze_perceiver False \
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
        --save_steps $SAVE_STEPS \
        --save_total_limit 2 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing $GRADIENT_CKPT \
        --logging_steps 10 \
        --model_max_length $MODEL_MAX_LENGTH \
        --report_to wandb \
        --run_name $WANDB_RUN_NAME \
        --dataloader_num_workers $DATALOADER_WORKERS \
        --box_score_thres 0.15 \
        --remove_unused_columns False \
        --dataloader_pin_memory True \
        | tee $OUTPUT_DIR/train.log
else
    # Multi-GPU training with torchrun
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=25003 \
        groma/train/train_mem.py \
        --model_name_or_path $MODEL_PATH \
        --dataset_config groma/data/configs/hoi_finetune.py \
        --freeze_perceiver False \
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
        --save_steps $SAVE_STEPS \
        --save_total_limit 2 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing $GRADIENT_CKPT \
        --logging_steps 10 \
        --model_max_length $MODEL_MAX_LENGTH \
        --report_to wandb \
        --run_name $WANDB_RUN_NAME \
        --dataloader_num_workers $DATALOADER_WORKERS \
        --box_score_thres 0.15 \
        --remove_unused_columns False \
        --dataloader_pin_memory True \
        | tee $OUTPUT_DIR/train.log
fi

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
