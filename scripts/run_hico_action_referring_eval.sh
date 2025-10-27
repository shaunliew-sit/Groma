#!/bin/bash
################################################################################
# HICO-DET Action Referring Task Evaluation Script
# Evaluates HOI action prediction using METEOR and CIDEr metrics
#
# Task: Given (person, object) bounding boxes, predict the action connecting them
# Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
#
# Usage:
#   bash scripts/run_hico_action_referring_eval.sh [GPU] [MODEL] [ANN_FILE] [IMAGES_DIR] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_hico_action_referring_eval.sh 0
#
#   # With debugging flags
#   VERBOSE=1 bash scripts/run_hico_action_referring_eval.sh 0
#   MAX_IMAGES=10 bash scripts/run_hico_action_referring_eval.sh 0
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_action_referring_eval.sh 0
#
#   # With W&B logging
#   WANDB=1 bash scripts/run_hico_action_referring_eval.sh 0
#   WANDB=1 WANDB_PROJECT=hoi-eval WANDB_RUN_NAME=hico-action-v1 \
#       bash scripts/run_hico_action_referring_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1         Show per-triplet results + action visualizations
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: hico-action-referring)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/hico_action_predictions_{timestamp}.json         # Raw predictions
#   {output_dir}/hico_action_predictions_{timestamp}_per_triplet.json  # Detailed per-triplet (VERBOSE)
#   {output_dir}/hico_action_predictions_{timestamp}_per_action.json   # Per-action breakdown (VERBOSE)
#   {output_dir}/hico_action_predictions_{timestamp}_metrics.json      # METEOR/CIDEr scores
#   {output_dir}/hico_action_evaluation_{timestamp}.log                # Full log
#   {output_dir}/visualizations_{timestamp}/*.jpg                      # Visualizations (VERBOSE)
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-groma-7b-finetune}"
HICO_ANN="${3:-../data/hico_20160224_det/annotations/test_hico_ann.json}"
IMAGES_DIR="${4:-../data/hico_20160224_det/images/test2015}"
OUTPUT_DIR="${5:-results/hico_action_referring}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_evaluation_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/hico_action_predictions_${TIMESTAMP}.json"
METRICS_FILE="$OUTPUT_DIR/hico_action_predictions_${TIMESTAMP}_metrics.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (METEOR & CIDEr)"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Annotation:  $HICO_ANN"
echo "Images:      $IMAGES_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Pred file:   $PRED_FILE"
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

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
WANDB_PROJECT_FLAG=""
WANDB_RUN_NAME_FLAG=""

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

    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_PROJECT_FLAG="--wandb-project $WANDB_PROJECT"
        echo "  Project: $WANDB_PROJECT"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_RUN_NAME_FLAG="--wandb-run-name $WANDB_RUN_NAME"
        echo "  Run: $WANDB_RUN_NAME"
    fi
fi

echo ""

# Prepare COCO-format dataset (if not exists)
COCO_ANN_FILE="groma_data/hico_action_referring_test.json"
GROMA_INSTRUCT_FILE="../data/hico_20160224_det/annotations/groma_instructions/latest_hico_grounding_referring_test.json"

if [ ! -f "$COCO_ANN_FILE" ]; then
    echo "========================================================================"
    echo "Preparing COCO-format dataset from Groma instructions..."
    echo "========================================================================"

    # Check if Groma instruction file exists, if not generate it
    if [ ! -f "$GROMA_INSTRUCT_FILE" ]; then
        echo "Generating Groma instruction format test data..."
        python3 groma/data/annotation_converters/hico_to_instruct.py \
            --anno-file "$HICO_ANN" \
            --output-file "$GROMA_INSTRUCT_FILE" \
            --num-variations 1
        echo ""
    fi

    # Convert to COCO format
    python scripts/prepare_action_referring_from_groma_instruct.py \
        --input-file "$GROMA_INSTRUCT_FILE" \
        --output-file "$COCO_ANN_FILE"
    echo ""
    echo "✓ Dataset prepared: $COCO_ANN_FILE"
    echo ""
else
    echo "✓ COCO-format dataset already exists: $COCO_ANN_FILE"
    echo ""
fi

# Run evaluation
echo "========================================================================"
echo "Starting evaluation..."
echo "========================================================================"
echo ""

python groma/eval/eval_hico_action_referring.py \
    --model-name "$MODEL_PATH" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$COCO_ANN_FILE" \
    --pred-file "$PRED_FILE" \
    --metrics-file "$METRICS_FILE" \
    --log-file "$LOG_FILE" \
    --batch-size 1 \
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
    echo "  Predictions:  $PRED_FILE"
    echo "  Metrics:      $METRICS_FILE"
    echo "  Log:          $LOG_FILE"
    echo ""

    if [ ! -z "$VERBOSE" ]; then
        PER_TRIPLET_FILE="${PRED_FILE//.json/_per_triplet.json}"
        PER_ACTION_FILE="${PRED_FILE//.json/_per_action.json}"
        VIZ_DIR="${OUTPUT_DIR}/visualizations_${TIMESTAMP}"

        echo "Verbose outputs:"
        echo "  Per-triplet:  $PER_TRIPLET_FILE"
        echo "  Per-action:   $PER_ACTION_FILE"
        echo "  Visualizations: $VIZ_DIR/"
        echo ""
    fi

    echo "View metrics:"
    echo "  cat $METRICS_FILE | jq '.'"
    echo ""
    echo "Key metrics:"
    echo "  METEOR:   Semantic similarity (0-100%, higher=better)"
    echo "  CIDEr:    Corpus consensus (0-200%+, higher=better)"
    echo "  BLEU:     N-gram overlap (0-100%, higher=better)"
    echo "  ROUGE-L:  Longest common subsequence (0-100%, higher=better)"
    echo ""
    echo "Metric interpretation:"
    echo "  METEOR 80%+:    Excellent action prediction"
    echo "  METEOR 70-80%:  Good action prediction"
    echo "  METEOR 60-70%:  Fair action prediction"
    echo "  METEOR <60%:    Poor action prediction"
    echo ""
    echo "  CIDEr 100%+:    Very good corpus alignment"
    echo "  CIDEr 50-100%:  Moderate corpus alignment"
    echo "  CIDEr <50%:     Poor corpus alignment"
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
