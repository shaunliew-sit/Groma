#!/bin/bash
################################################################################
# SWIG-HOI Grounding Task Evaluation Script for InternVL3
# Evaluates InternVL3 grounding performance on SWIG-HOI dataset
# Each sample = one (action, object) combination with multi-pair support
# Supports person-person interactions
#
# Task: Detect person-object pairs with bounding boxes
# Metrics: COCO-style AR (Average Recall)
#
# Usage:
#   bash scripts/run_swig_ground_eval_internvl3.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage with different GPUs
#   bash scripts/run_swig_ground_eval_internvl3.sh 0                    # Use GPU 0
#   bash scripts/run_swig_ground_eval_internvl3.sh 1                    # Use GPU 1
#   bash scripts/run_swig_ground_eval_internvl3.sh cuda:0               # Explicit cuda:0
#
#   # With optional flags (set as environment variables)
#   VERBOSE=1 bash scripts/run_swig_ground_eval_internvl3.sh 0          # Show per-image results
#   MAX_IMAGES=10 bash scripts/run_swig_ground_eval_internvl3.sh 0      # Test on first 10 images
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_ground_eval_internvl3.sh 1  # Both flags
#
#   # With Weights & Biases logging
#   WANDB=1 VERBOSE=1 bash scripts/run_swig_ground_eval_internvl3.sh 1            # Enable WandB
#   WANDB=1 WANDB_PROJECT="internvl3-swig" bash scripts/run_swig_ground_eval_internvl3.sh 0
#
#   # Use different model size
#   bash scripts/run_swig_ground_eval_internvl3.sh 0 "OpenGVLab/InternVL3-4B"
#
# Environment Variables:
#   VERBOSE=1         Show per-image results
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: swig-grounding-internvl3)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/swig_ground_internvl3_results_{timestamp}.json          # Raw predictions
#   {output_dir}/swig_ground_internvl3_results_{timestamp}_metrics.json # AR metrics
#   {output_dir}/swig_ground_internvl3_results_{timestamp}_action_stats.json # Per-action
#   {output_dir}/swig_ground_internvl3_evaluation_{timestamp}.log        # Full log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_NAME="${2:-OpenGVLab/InternVL3-38B}"
OUTPUT_DIR="${3:-results-redo/swig_ground_internvl3-38B}"

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
LOG_FILE="$OUTPUT_DIR/swig_ground_internvl3_evaluation_${TIMESTAMP}.log"

# SWIG dataset paths
SWIG_ROOT="../data/swig_hoi"
IMG_PREFIX="${SWIG_ROOT}/images_512"
ANN_FILE="groma_data/benchmarks/swig_ground_test.json"
RESULT_FILE="${OUTPUT_DIR}/swig_ground_internvl3_results_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "SWIG-HOI Grounding Evaluation (InternVL3)"
echo "========================================================================"
echo "GPU:         $GPU_ID (Device: $DEVICE_ARG)"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
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
    echo "Please check the path to SWIG images_512"
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
    echo "✓ Verbose mode enabled (per-image results)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES images"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    # Optional WandB project and run name
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: swig-grounding-internvl3 (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 groma/eval/eval_swig_ground_internvl3.py \
    --model-name \"$MODEL_NAME\" \
    --device $DEVICE_ARG \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE"

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
    echo "  Predictions:    $RESULT_FILE"
    echo "  Metrics:        ${RESULT_FILE//.json/_metrics.json}"
    echo "  Action stats:   ${RESULT_FILE//.json/_action_stats.json}"
    echo "  Log:            $LOG_FILE"
    echo ""

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
    echo ""
    echo "Note: This dataset includes person-person interactions"
    echo "Compare with Groma and Qwen3VL baselines:"
    echo "  See results/swig_ground/ for Groma results"
    echo "  See results/swig_ground_qwen3vl/ for Qwen3VL results"
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
