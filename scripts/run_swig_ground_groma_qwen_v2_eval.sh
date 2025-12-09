#!/bin/bash
################################################################################
# [GROMA-QWEN V2] SWIG-HOI Grounding Task Evaluation Script (JSON Format)
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# Evaluates Groma-Qwen V2 grounding performance using JSON bbox format
# aligned with Qwen3-VL pre-trained grounding capabilities.
#
# Each sample = one (action, object) combination with multi-pair support
#
# Task: Given action + object description, predict bounding boxes for ALL
#       person-object pairs in JSON format: [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]
#
# Metrics: COCO-style AR (Average Recall) with size-specific metrics
#
# Usage:
#   bash scripts/run_swig_ground_groma_qwen_v2_eval.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0
#   bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 1
#
#   # With optional flags (set as environment variables)
#   VERBOSE=1 bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0
#   MAX_IMAGES=10 bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 7
#
#   # With Weights & Biases logging
#   WANDB=1 VERBOSE=1 bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 4
#   WANDB=1 WANDB_PROJECT="my-hoi-eval" bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0
#   WANDB=1 WANDB_PROJECT="my-project" WANDB_RUN_NAME="test-run" bash scripts/run_swig_ground_groma_qwen_v2_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1             Show per-image results + visualizations
#   MAX_IMAGES=N          Limit to first N images (for quick testing)
#   WANDB=1               Enable Weights & Biases logging
#   WANDB_PROJECT=name    W&B project name (default: groma-qwen-v2-grounding)
#   WANDB_RUN_NAME=name   W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/swig_ground_v2_{timestamp}.json                    # Raw predictions
#   {output_dir}/swig_ground_v2_{timestamp}_per_image.json          # Detailed (VERBOSE)
#   {output_dir}/swig_ground_v2_{timestamp}_metrics.json            # AR metrics
#   {output_dir}/swig_ground_v2_{timestamp}_action_stats.json       # Per-action stats
#   {output_dir}/swig_ground_v2_{timestamp}.log                     # Full log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-qwen-v2-stage2-grounding-json}"
BASE_MODEL_PATH="${3:-checkpoints/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${4:-results-groma-qwen/swig-qwen-v2-stage2-grounding-json}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/swig_ground_v2_${TIMESTAMP}.log"

# SWIG dataset paths
SWIG_ROOT="../data/swig_hoi"
IMG_PREFIX="${SWIG_ROOT}/images_512"
ANN_FILE="groma_data/benchmarks/swig_ground_test_json.json"
RESULT_FILE="${OUTPUT_DIR}/swig_ground_v2_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "[GROMA-QWEN V2] SWIG-HOI Grounding Evaluation (JSON Format)"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Base Model:  $BASE_MODEL_PATH"
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
    echo "Please check the path to SWIG images"
    exit 1
fi

# Count number of test images
NUM_IMAGES=$(ls -1 "$IMG_PREFIX"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
WANDB_PROJECT_FLAG=""
WANDB_RUN_NAME_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-image results + visualizations)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES samples (HOI triplets)"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_PROJECT_FLAG="--wandb-project $WANDB_PROJECT"
        echo "  Project: $WANDB_PROJECT"
    else
        echo "  Project: groma-qwen-v2-grounding (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_RUN_NAME_FLAG="--wandb-run-name $WANDB_RUN_NAME"
        echo "  Run name: $WANDB_RUN_NAME"
    fi
fi

echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation (using same Python script as HICO, just different annotation)
# Note: Use --device 0 because CUDA_VISIBLE_DEVICES remaps the selected GPU to device 0
python groma/eval/eval_hico_ground_groma_qwen_v2.py \
    --model-name "$MODEL_PATH" \
    --base-model-name "$BASE_MODEL_PATH" \
    --device 0 \
    --ann-file "$ANN_FILE" \
    --img-prefix "$IMG_PREFIX" \
    --result-file "$RESULT_FILE" \
    $VERBOSE_FLAG \
    $MAX_IMAGES_FLAG \
    $WANDB_FLAG \
    $WANDB_PROJECT_FLAG \
    $WANDB_RUN_NAME_FLAG \
    2>&1 | tee "$LOG_FILE"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Results saved to:"
    echo "  Predictions:  $RESULT_FILE"
    echo "  Metrics:      ${RESULT_FILE//.json/_metrics.json}"
    echo "  Action stats: ${RESULT_FILE//.json/_action_stats.json}"
    echo "  Log:          $LOG_FILE"
    echo ""

    if [ ! -z "$VERBOSE" ]; then
        PER_IMAGE_FILE="${RESULT_FILE//.json/_per_image.json}"
        VIZ_DIR="${RESULT_FILE//.json/_visualizations}"

        echo "Verbose outputs:"
        echo "  Per-image:      $PER_IMAGE_FILE"
        echo "  Visualizations: $VIZ_DIR/"
        echo ""
    fi

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
    echo ""
    echo "Size-specific metrics:"
    echo "  ARs: Small objects (area < 32^2 = 1024 pixels²)"
    echo "  ARm: Medium objects (32^2 <= area < 96^2 = 9216 pixels²)"
    echo "  ARl: Large objects (area >= 96^2 = 9216 pixels²)"
    echo ""
    echo "View metrics:"
    echo "  cat ${RESULT_FILE//.json/_metrics.json} | jq '.'"
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

