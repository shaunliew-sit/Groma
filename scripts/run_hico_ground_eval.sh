#!/bin/bash
################################################################################
# HICO-DET Object Grounding Evaluation Script
# Evaluates object grounding performance on HICO-DET dataset
#
# Task: Given object category names, predict bounding boxes for those objects
# Metrics: COCO-style AP/AR (similar to LVIS-Ground)
#
# Usage:
#   bash scripts/run_hico_ground_eval.sh [GPU] [MODEL] [ANN_FILE] [IMAGES_DIR] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_hico_ground_eval.sh 0                    # Use GPU 0 with defaults
#   bash scripts/run_hico_ground_eval.sh 1                    # Use GPU 1 with defaults
#   bash scripts/run_hico_ground_eval.sh 0,1                  # Use GPUs 0 and 1
#   bash scripts/run_hico_ground_eval.sh 2 /path/to/model     # Use GPU 2 with custom model
#
#   # With optional flags (set as environment variables)
#   VERBOSE=1 bash scripts/run_hico_ground_eval.sh 0          # Show per-image results
#   MAX_IMAGES=10 bash scripts/run_hico_ground_eval.sh 0      # Test on first 10 images
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_ground_eval.sh 0  # Both flags
#
# Output files:
#   - {output_dir}/hico_ground_predictions_{timestamp}.json           # Predictions
#   - {output_dir}/hico_ground_predictions_{timestamp}_per_image_results.json  # Detailed per-image results
#   - {output_dir}/hico_ground_evaluation_{timestamp}.log             # Full log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-groma-7b-finetune}"
HICO_ANN="${3:-../data/hico_20160224_det/annotations/test_hico_ann.json}"
IMAGES_DIR="${4:-../data/hico_20160224_det/images/test2015}"
OUTPUT_DIR="${5:-results/hico_ground}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_evaluation_${TIMESTAMP}.log"
RESULT_FILE="$OUTPUT_DIR/hico_ground_predictions_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Object Grounding Evaluation"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Annotation:  $HICO_ANN"
echo "Images:      $IMAGES_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download the model:"
    echo "  git clone https://huggingface.co/FoundationVision/groma-7b-finetune $MODEL_PATH"
    exit 1
fi

if [ ! -f "$HICO_ANN" ]; then
    echo "ERROR: Annotation file not found at $HICO_ANN"
    echo "Please check the path to test_hico_ann.json"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    echo "Please check the path to HICO test2015 images"
    exit 1
fi

# Count number of test images
NUM_IMAGES=$(ls -1 "$IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

echo "Starting evaluation..."
echo ""

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "Verbose mode enabled"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "Limiting to first $MAX_IMAGES images"
fi

# Run evaluation
python groma/eval/eval_hico_ground.py \
    --model-name "$MODEL_PATH" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$HICO_ANN" \
    --result-file "$RESULT_FILE" \
    --box_score_thres 0.15 \
    --batch_size_per_gpu 1 \
    $VERBOSE_FLAG \
    $MAX_IMAGES_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Evaluation complete!"
echo "========================================================================"
echo "Results saved to:"
echo "  Predictions:  ${RESULT_FILE}"
echo "  Ground truth: ${RESULT_FILE//_predictions_/_ground_truth_}"
echo "  Metrics:      ${RESULT_FILE//.json/_metrics.json}"
echo "  Log:          ${LOG_FILE}"
echo ""
echo "Key metrics (from COCO evaluation):"
echo "  AP:      Average Precision @ IoU=0.50:0.95"
echo "  AP50:    Average Precision @ IoU=0.50"
echo "  AP75:    Average Precision @ IoU=0.75"
echo "  AR:      Average Recall @ IoU=0.50:0.95"
echo ""
echo "Size-specific metrics:"
echo "  APs/ARs: Small objects (area < 32^2)"
echo "  APm/ARm: Medium objects (32^2 < area < 96^2)"
echo "  APl/ARl: Large objects (area > 96^2)"
echo "========================================================================"
