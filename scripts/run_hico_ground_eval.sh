#!/bin/bash
################################################################################
# HICO-DET Grounding Task Evaluation Script
# Evaluates grounding performance on HICO-DET dataset
# Each sample = one (action, object) combination with multi-pair support
#
# Task: Detect entities and describe HOI interactions with bounding boxes
# Metrics: COCO-style AR (Average Recall)
#
# Usage:
#   bash scripts/run_hico_ground_eval.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_hico_ground_eval.sh 0                    # Use GPU 0 with defaults
#   bash scripts/run_hico_ground_eval.sh 1                    # Use GPU 1 with defaults
#   bash scripts/run_hico_ground_eval.sh 0,1                  # Use GPUs 0 and 1
#
#   # With optional flags (set as environment variables)
#   VERBOSE=1 bash scripts/run_hico_ground_eval.sh 0          # Show per-image results + visualizations
#   MAX_IMAGES=10 bash scripts/run_hico_ground_eval.sh 0      # Test on first 10 images
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_ground_eval.sh 0  # Both flags
#
#   # With Weights & Biases logging
#   WANDB=1 bash scripts/run_hico_ground_eval.sh 0            # Enable WandB logging
#   WANDB=1 VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_ground_eval.sh 0  # Full debug + WandB
#   WANDB=1 WANDB_PROJECT="my-hoi-eval" WANDB_RUN_NAME="test-run" bash scripts/run_hico_ground_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1         Show per-image results + visualizations
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: hico-grounding)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/hico_ground_results_{timestamp}.json                    # Raw predictions
#   {output_dir}/hico_ground_results_{timestamp}_per_image.json         # Detailed per-image (VERBOSE)
#   {output_dir}/hico_ground_results_{timestamp}_per_action.json        # Per-action breakdown (VERBOSE)
#   {output_dir}/hico_ground_results_{timestamp}_metrics.json           # AR metrics
#   {output_dir}/hico_ground_evaluation_{timestamp}.log                 # Full log
#   {output_dir}/visualizations_{timestamp}/*.jpg                       # Visualizations (VERBOSE)
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-7b-finetune-hoi-v2}"
OUTPUT_DIR="${3:-results/hico_ground}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_evaluation_${TIMESTAMP}.log"

# HICO dataset paths
HICO_ROOT="/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det"
IMG_PREFIX="${HICO_ROOT}/images/test2015"
ANN_FILE="groma_data/benchmarks/hico_ground_test.json"
RESULT_FILE="${OUTPUT_DIR}/hico_ground_results_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Grounding Evaluation"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please provide a valid model path"
    exit 1
fi

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
    echo "✓ Verbose mode enabled (per-image results + visualizations)"
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
        echo "  WandB project: hico-grounding (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

echo ""

# Run evaluation
# Note: Don't use trailing backslashes after variable flags to avoid shell parsing errors
EVAL_CMD="python3 groma/eval/eval_hico_ground.py \
    --model-name $MODEL_PATH \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE \
    --box-score-thres 0.15"

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
    echo "  Predictions:  $RESULT_FILE"
    echo "  Metrics:      ${RESULT_FILE//.json/_metrics.json}"
    echo "  Log:          $LOG_FILE"
    echo ""

    if [ ! -z "$VERBOSE" ]; then
        PER_IMAGE_FILE="${RESULT_FILE//.json/_per_image.json}"
        PER_ACTION_FILE="${RESULT_FILE//.json/_per_action.json}"
        VIZ_DIR="${OUTPUT_DIR}/visualizations_${TIMESTAMP}"

        echo "Verbose outputs:"
        echo "  Per-image:      $PER_IMAGE_FILE"
        echo "  Per-action:     $PER_ACTION_FILE"
        echo "  Visualizations: $VIZ_DIR/"
        echo ""
    fi

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
    echo ""
    echo "Size-specific metrics:"
    echo "  ARs: Small objects (area < 32^2)"
    echo "  ARm: Medium objects (32^2 < area < 96^2)"
    echo "  ARl: Large objects (area > 96^2)"
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
