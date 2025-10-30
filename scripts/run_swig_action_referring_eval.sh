#!/bin/bash
################################################################################
# SWIG-HOI Action Referring Task Evaluation Script
# Evaluates HOI action prediction using METEOR and CIDEr metrics
#
# Task: Given (person, object) bounding boxes, predict the action connecting them
# Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
#
# Note: SWIG actions are in -ing form (e.g., "stapling", "stirring") vs HICO root form
#
# Usage:
#   bash scripts/run_swig_action_referring_eval.sh [GPU] [MODEL] [ANN_FILE] [IMAGES_DIR] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_swig_action_referring_eval.sh 0
#
#   # With debugging flags
#   VERBOSE=1 bash scripts/run_swig_action_referring_eval.sh 0
#   MAX_IMAGES=10 bash scripts/run_swig_action_referring_eval.sh 0
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_action_referring_eval.sh 0
#
#   # With W&B logging
#   WANDB=1 bash scripts/run_swig_action_referring_eval.sh 0
#   WANDB=1 WANDB_PROJECT=hoi-eval WANDB_RUN_NAME=swig-action-v1 \
#       bash scripts/run_swig_action_referring_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1         Show per-triplet results + action visualizations
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: swig-action-referring)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/swig_action_predictions_{timestamp}.json         # Raw predictions
#   {output_dir}/swig_action_predictions_{timestamp}_per_triplet.json  # Detailed per-triplet (VERBOSE)
#   {output_dir}/swig_action_predictions_{timestamp}_per_action.json   # Per-action breakdown (VERBOSE)
#   {output_dir}/swig_action_predictions_{timestamp}_metrics.json      # METEOR/CIDEr scores
#   {output_dir}/swig_action_evaluation_{timestamp}.log                # Full log
#   {output_dir}/visualizations_{timestamp}/*.jpg                      # Visualizations (VERBOSE)
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-7b-finetune-hoi-v2}"
SWIG_ROOT="/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi"
BENCHMARK_ANN="${3:-groma_data/benchmarks/swig_action_referring_test_coco.json}"
IMAGES_DIR="${4:-${SWIG_ROOT}/images_512}"
OUTPUT_DIR="${5:-results/swig_action_referring}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/swig_action_evaluation_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/swig_action_predictions_${TIMESTAMP}.json"
METRICS_FILE="$OUTPUT_DIR/swig_action_predictions_${TIMESTAMP}_metrics.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "SWIG-HOI Action Referring Evaluation (METEOR & CIDEr)"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Annotation:  $BENCHMARK_ANN"
echo "Images:      $IMAGES_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Pred file:   $PRED_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please ensure the model checkpoint is available"
    exit 1
fi

if [ ! -f "$BENCHMARK_ANN" ]; then
    echo "ERROR: Benchmark annotation file not found at $BENCHMARK_ANN"
    echo "Please generate it using:"
    echo "  python3 groma/data/annotation_converters/swig_to_action_referring.py"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    echo "Please check the path to SWIG images_512"
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

# Run evaluation
echo "========================================================================"
echo "Starting evaluation..."
echo "========================================================================"
echo ""

python groma/eval/eval_swig_action_referring.py \
    --model-name "$MODEL_PATH" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$BENCHMARK_ANN" \
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
