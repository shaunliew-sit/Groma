#!/bin/bash
################################################################################
# HICO-DET Action Referring Task Evaluation Script for Qwen3VL
# Evaluates Qwen3VL action prediction using METEOR and CIDEr metrics
#
# Task: Given (person, object) bounding boxes, predict the connecting action
# Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
#
# Usage:
#   bash scripts/run_hico_action_referring_eval_qwen3vl.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage with different GPUs
#   bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0                    # Use GPU 0
#   bash scripts/run_hico_action_referring_eval_qwen3vl.sh 1                    # Use GPU 1
#   bash scripts/run_hico_action_referring_eval_qwen3vl.sh cuda:0               # Explicit cuda:0
#
#   # With optional flags (set as environment variables)
#   VERBOSE=1 bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0          # Show per-triplet results
#   MAX_IMAGES=10 bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0      # Test on first 10 triplets
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0  # Both flags
#
#   # With Weights & Biases logging
#   WANDB=1 bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0            # Enable WandB
#   WANDB=1 WANDB_PROJECT="qwen3vl-hico-action" bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0
#
#   # Use different model size
#   bash scripts/run_hico_action_referring_eval_qwen3vl.sh 0 "Qwen/Qwen3-VL-32B-Instruct"
#
# Environment Variables:
#   VERBOSE=1         Show per-triplet results + action visualizations
#   MAX_IMAGES=N      Limit to first N triplets (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: hico-action-referring-qwen3vl)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/hico_action_qwen3vl_results_{timestamp}.json          # Raw predictions
#   {output_dir}/hico_action_qwen3vl_results_{timestamp}_metrics.json # METEOR/CIDEr scores
#   {output_dir}/hico_action_qwen3vl_results_{timestamp}_per_triplet.json  # Detailed per-triplet (VERBOSE)
#   {output_dir}/hico_action_qwen3vl_results_{timestamp}_per_action.json   # Per-action breakdown (VERBOSE)
#   {output_dir}/hico_action_qwen3vl_evaluation_{timestamp}.log        # Full log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_NAME="${2:-Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${3:-results/hico_action_qwen3vl}"

# Set GPU (handle both "0" and "cuda:0" formats)
if [[ "$GPU_ID" == cuda:* ]]; then
    # Already in cuda:X format, use as-is for device argument
    DEVICE_ARG="$GPU_ID"
    # Extract just the number for CUDA_VISIBLE_DEVICES
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    # Just a number, use cuda:X format
    DEVICE_ARG="cuda:$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_qwen3vl_evaluation_${TIMESTAMP}.log"

# HICO dataset paths
HICO_ROOT="/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det"
IMG_PREFIX="${HICO_ROOT}/images/test2015"
ANN_FILE="groma_data/benchmarks/hico_action_referring_test.json"
PRED_FILE="${OUTPUT_DIR}/hico_action_qwen3vl_results_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (Qwen3VL)"
echo "========================================================================"
echo "GPU:         $GPU_ID (Device: $DEVICE_ARG)"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Pred file:   $PRED_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found at $ANN_FILE"
    echo "Please ensure the benchmark file has been generated"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found at $IMG_PREFIX"
    echo "Please check the path to HICO test2015 images"
    exit 1
fi

# Count number of test images
NUM_IMAGES=$(ls -1 "$IMG_PREFIX"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

echo "Starting evaluation..."
echo ""

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-triplet results + visualizations)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES triplets"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    # Optional WandB project and run name
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: hico-action-referring-qwen3vl (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 groma/eval/eval_hico_action_referring_qwen3vl.py \
    --model-name \"$MODEL_NAME\" \
    --device $DEVICE_ARG \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --pred-file $PRED_FILE"

# Add optional flags if they exist
if [ ! -z "$VERBOSE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
fi

if [ ! -z "$MAX_IMAGES_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
fi

if [ ! -z "$WANDB_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $WANDB_FLAG"
fi

# Execute the command
eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Results saved to:"
    echo "  Predictions:  $PRED_FILE"
    echo "  Metrics:      ${PRED_FILE//.json/_metrics.json}"
    echo "  Log:          $LOG_FILE"
    echo ""

    if [ ! -z "$VERBOSE" ]; then
        PER_TRIPLET_FILE="${PRED_FILE//.json/_per_triplet.json}"
        PER_ACTION_FILE="${PRED_FILE//.json/_per_action.json}"
        VIZ_DIR="${PRED_FILE//.json/_visualizations}"

        echo "Verbose outputs:"
        echo "  Per-triplet:  $PER_TRIPLET_FILE"
        echo "  Per-action:   $PER_ACTION_FILE"
        echo "  Visualizations: $VIZ_DIR/"
        echo ""
    fi

    echo "Key metrics:"
    echo "  METEOR:   Semantic similarity (0-100%, higher=better)"
    echo "  CIDEr:    Corpus consensus (0-200%+, higher=better)"
    echo "  BLEU:     N-gram overlap (0-100%, higher=better)"
    echo "  ROUGE-L:  Longest common subsequence (0-100%, higher=better)"
    echo ""
    echo "Compare with Groma baseline:"
    echo "  See results/hico_action_referring/ for Groma results"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Evaluation failed!"
    echo "========================================================================"
    echo "Check the log file for details:"
    echo "  $LOG_FILE"
    echo "========================================================================"
    exit 1
fi
