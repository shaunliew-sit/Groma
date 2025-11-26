#!/bin/bash
################################################################################
# [GROMA-QWEN] HICO-DET Action Referring Task Evaluation Script for Groma Qwen3VL
# Part of: Groma Qwen3VL Referring Task Implementation
# Evaluates HOI action prediction using METEOR and CIDEr metrics
#
# Task: Given (person, object) bounding boxes, predict the action connecting them
# Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
#
# Usage:
#   bash scripts/run_hico_action_referring_groma_qwen_eval.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage
#   bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
#
#   # With debugging flags
#   VERBOSE=1 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
#   MAX_IMAGES=10 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 5
#
#   # With W&B logging
#   WANDB=1 VERBOSE=1 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 5
#   WANDB=1 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
#   WANDB=1 WANDB_PROJECT=hoi-eval WANDB_RUN_NAME=hico-action-v1 \
#       bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1         Show per-triplet results + action visualizations with METEOR/CIDEr scores
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: hico-action-referring-groma-qwen)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#   USE_LEGACY_FORMAT=1  Use legacy instruction format instead of new Qwen3VL-native format
#
# Output files:
#   {output_dir}/hico_action_groma_qwen_results_{timestamp}.json         # Raw predictions
#   {output_dir}/hico_action_groma_qwen_results_{timestamp}_per_triplet.json  # Detailed per-triplet (VERBOSE) with METEOR/CIDEr scores
#   {output_dir}/hico_action_groma_qwen_results_{timestamp}_per_action.json   # Per-action breakdown (VERBOSE)
#   {output_dir}/hico_action_groma_qwen_results_{timestamp}_metrics.json      # METEOR/CIDEr scores
#   {output_dir}/hico_action_groma_qwen_evaluation_{timestamp}.log                # Full log
#   {output_dir}/visualizations_{timestamp}/*.jpg                      # Visualizations (VERBOSE) with scores
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-qwen-stage3-referring-only-24-nov-fixed}"
HICO_ROOT="../data/hico_20160224_det"
BENCHMARK_ANN="${3:-groma_data/benchmarks/hico_action_referring_test.json}"
IMAGES_DIR="${4:-${HICO_ROOT}/images/test2015}"
OUTPUT_DIR="${5:-results-redo-groma/hico_action_referring_groma_qwen}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_groma_qwen_evaluation_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/hico_action_groma_qwen_results_${TIMESTAMP}.json"
METRICS_FILE="$OUTPUT_DIR/hico_action_groma_qwen_results_${TIMESTAMP}_metrics.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (Groma Qwen3VL) - METEOR & CIDEr"
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
    echo "Please ensure the benchmark file has been generated"
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
FORMAT_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-triplet results + visualizations with METEOR/CIDEr scores)"
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

if [ ! -z "$USE_LEGACY_FORMAT" ]; then
    FORMAT_FLAG="--use-legacy-format"
    echo "✓ Using LEGACY instruction format"
else
    echo "✓ Using NEW Qwen3VL-native instruction format (default)"
fi

echo ""

# Run evaluation
echo "========================================================================"
echo "Starting evaluation..."
echo "========================================================================"
echo ""

python3 groma/eval/eval_hico_action_referring_groma_qwen.py \
    --model-name "$MODEL_PATH" \
    --device "$GPU_ID" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$BENCHMARK_ANN" \
    --pred-file "$PRED_FILE" \
    $VERBOSE_FLAG \
    $MAX_IMAGES_FLAG \
    $WANDB_FLAG \
    $WANDB_PROJECT_FLAG \
    $WANDB_RUN_NAME_FLAG \
    $FORMAT_FLAG \
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
        echo "  Per-triplet:  $PER_TRIPLET_FILE (includes METEOR/CIDEr scores for each triplet)"
        echo "  Per-action:   $PER_ACTION_FILE"
        echo "  Visualizations: $VIZ_DIR/ (with METEOR/CIDEr scores displayed)"
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
    if [ ! -z "$VERBOSE" ]; then
        echo "In verbose mode, METEOR and CIDEr scores are shown for each triplet/image"
        echo ""
    fi
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

