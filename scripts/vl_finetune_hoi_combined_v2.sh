#!/bin/bash
# Stage 3: Instruction Finetuning - HICO + SWIG Combined (Grounding + Action Referring)
#
# This script trains using COMBINED datasets with interleaved grounding and referring tasks.
# Uses the CURRENT format (same as separate files) but with task interleaving benefits.
#
# Key Improvements:
# - Task interleaving: Model sees grounding + referring for same image
# - Same format: Exact same prompts/responses as separate files
# - Better multi-task learning: Natural task switching
#
# Usage:
#   bash scripts/vl_finetune_hoi_combined_v2.sh <PRETRAIN_PATH> <OUTPUT_DIR> [GPU_IDS] [NUM_GPUS] [RUN_NAME]
#
# Arguments:
#   PRETRAIN_PATH : Path to Groma-7b-pretrain checkpoint from Stage 2 (required)
#   OUTPUT_DIR    : Output directory (required)
#   GPU_IDS       : Comma-separated GPU IDs (e.g., "5,6,7") [default: "0,1,2,3,4,5,6,7"]
#   NUM_GPUS      : Number of GPUs (overrides GPU_IDS if set) [default: 8]
#   RUN_NAME      : WandB run name [default: "vl-finetune-hoi-combined-v2-<timestamp>"]
#
# Examples:
#   # Use GPUs 5,6,7
#   bash scripts/vl_finetune_hoi_combined_v2.sh \
#       groma-7b-finetune \
#       checkpoints/groma-7b-hoi-ground-refer-fine-tuned-hungarian-mixed-v7 \
#       "1,7"
#
#   # Use first 4 GPUs
#   bash scripts/vl_finetune_hoi_combined_v2.sh \
#       checkpoints/groma-7b-pretrain-hoi \
#       checkpoints/groma-7b-finetune-hoi-combined-v2 \
#       "" \
#       4

# Required arguments
PRETRAIN_PATH=$1
OUTPUT_DIR=$2

# Optional arguments
GPU_IDS=${3:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${4:-8}
RUN_NAME=${5:-"vl-finetune-hoi-combined-v2-$(date +%Y%m%d-%H%M%S)"}

# Auto-calculate num GPUs from GPU_IDS if NUM_GPUS not explicitly set
if [ ! -z "$4" ] && [ "$4" != "" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
ACTUAL_NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l | tr -d ' ')

# WandB configuration
WANDB_PROJECT=${WANDB_PROJECT:-"groma-instruction-finetune-hoi-combined-v2"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Stage 3: Instruction Finetuning - HICO + SWIG COMBINED v2"
echo "            (Interleaved Grounding + Action Referring)"
echo "============================================================"
echo "Configuration:"
echo "  Pretrain Path: $PRETRAIN_PATH"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  GPU IDs:       $GPU_IDS"
echo "  Num GPUs:      $ACTUAL_NUM_GPUS"
echo ""
echo "Datasets (COMBINED FORMAT):"
echo "  HICO-DET Combined: 162,445 samples (69,404 grounding + 93,041 referring)"
echo "  SWIG-HOI Combined: 120,312 samples (59,519 grounding + 60,793 referring)"
echo "  Total:             282,757 samples"
echo ""
echo "Key Features:"
echo "  • INTERLEAVED TASKS: Grounding and referring for same image"
echo "  • SAME FORMAT: Exact format as separate files (proven to work)"
echo "  • Task switching: Better multi-task learning"
echo "  • Action + object output: 'riding motorcycle', 'talking person'"
echo ""
echo "Format Details:"
echo "  Grounding:"
echo "    Q: [grounding] Identify ... person {action} {object} and the {object}"
echo "    A: <p>person</p><roi><r0></roi><p>{object}</p><roi><r1></roi>"
echo ""
echo "  Action Referring:"
echo "    Q: What is <roi> <refer_box> </roi> ... doing with ... ?"
echo "    A: {action} {object}  (e.g., 'riding motorcycle')"
echo ""
echo "Training:"
echo "  Freeze LLM:        False  (LLM trainable)"
echo "  Freeze Perceiver:  True   (Detection head frozen)"
echo "  Trainable:         LLM + VL Bridge + Region Encoder"
echo "  Epochs:            1"
echo "  Learning Rate:     2e-5"
echo "  Batch Size:        $((16 * ACTUAL_NUM_GPUS)) (16 × $ACTUAL_NUM_GPUS GPUs)"
echo "  FSDP:              Enabled"
echo ""
echo "Expected Outcome:"
echo "  ✓ Restored performance (same format as working model)"
echo "  ✓ Improved multi-task learning (task interleaving)"
echo "  ✓ Consistent outputs (action + object format)"
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

# Verify combined benchmark files exist
BENCHMARK_DIR="groma_data/benchmarks"
REQUIRED_FILES=(
    "$BENCHMARK_DIR/hico_combined_ground_refer_train_v2.json"
    "$BENCHMARK_DIR/swig_combined_ground_refer_train_v2.json"
)

echo "Checking combined benchmark files..."
MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ❌ Missing: $file"
        MISSING_FILES=1
    else
        echo "  ✓ Found: $file"
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "❌ ERROR: Some combined benchmark files are missing!"
    echo ""
    echo "Please generate them using the combined converters:"
    echo "  python groma/data/annotation_converters/hico_to_combined.py \\"
    echo "      --anno-file /path/to/trainval_hico_ann.json \\"
    echo "      --output-file groma_data/benchmarks/hico_combined_ground_refer_train_v2.json"
    echo ""
    echo "  python groma/data/annotation_converters/swig_to_combined.py \\"
    echo "      --anno-file /path/to/swig_train_1000.json \\"
    echo "      --output-file groma_data/benchmarks/swig_combined_ground_refer_train_v2.json"
    exit 1
fi
echo "✓ All combined benchmark files present"
echo ""

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
    --model_name_or_path $PRETRAIN_PATH \
    --dataset_config groma/data/configs/vl_finetune_hoi_combined_v2.py \
    --freeze_perceiver True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_grad_norm 1.0 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config scripts/fsdp_config.json \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --model_max_length 2048 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataloader_num_workers 2 \
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
    echo "🎉 Combined Format Training Complete!"
    echo "============================================================"
    echo "Your HOI-specialized Groma model is ready for evaluation!"
    echo ""
    echo "Model trained with:"
    echo "  ✓ Interleaved grounding + referring tasks"
    echo "  ✓ Same format as working separate files"
    echo "  ✓ Action + object output format"
    echo "  ✓ 282,757 training samples"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate grounding on HICO-DET test set"
    echo "  2. Evaluate grounding on SWIG-HOI test set"
    echo "  3. Evaluate action referring on HICO-DET test set"
    echo "  4. Evaluate action referring on SWIG-HOI test set"
    echo ""
    echo "Evaluation commands:"
    echo "  # Grounding"
    echo "  bash scripts/run_hico_ground_eval.sh 0 $OUTPUT_DIR"
    echo "  bash scripts/run_swig_ground_eval.sh 0 $OUTPUT_DIR"
    echo ""
    echo "  # Action Referring"
    echo "  bash scripts/run_hico_action_referring_eval.sh 0 $OUTPUT_DIR"
    echo "  bash scripts/run_swig_action_referring_eval.sh 0 $OUTPUT_DIR"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed! Check: $OUTPUT_DIR/train.log"
    echo "============================================================"
    exit 1
fi
