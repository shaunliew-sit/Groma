#!/bin/bash
################################################################################
# [GROMA-QWEN V3] HICO-DET Action Referring Evaluation Script
# Part of: Groma Qwen3VL with Interaction Token Architecture
#
# Evaluates Groma-Qwen V2/V3 action prediction using METEOR and CIDEr metrics.
# Auto-detects V2 vs V3 models from config.json.
#
# Task: Given (person, object) bounding boxes, predict the action connecting them
# Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
#
# Usage:
#   bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh [GPU] [MODEL] [BASE_MODEL] [OUTPUT_DIR]
#
# Examples:
#   # V3 model evaluation
#   bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0 checkpoints/groma-qwen-v3-referring
#
#   # V2 model (auto-detected)
#   bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0 checkpoints/groma-qwen-v2-stage2
#
#   # With debugging flags
#   VERBOSE=1 bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0
#   MAX_IMAGES=10 bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0
#   VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0
#
#   # With W&B logging
#   WANDB=1 VERBOSE=1 bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0
#   WANDB=1 WANDB_PROJECT="my-project" bash scripts/run_hico_action_referring_groma_qwen_v3_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1             Show per-triplet results + visualizations with scores
#   MAX_IMAGES=N          Limit to first N images (for quick testing)
#   WANDB=1               Enable Weights & Biases logging
#   WANDB_PROJECT=name    W&B project name (default: groma-qwen-v3-eval)
#   WANDB_RUN_NAME=name   W&B run name (auto-generated if not provided)
#
# Output files:
#   {output_dir}/hico_action_v3_{timestamp}.json              # Raw predictions
#   {output_dir}/hico_action_v3_{timestamp}_per_triplet.json  # Per-triplet (VERBOSE)
#   {output_dir}/hico_action_v3_{timestamp}_per_action.json   # Per-action (VERBOSE)
#   {output_dir}/hico_action_v3_{timestamp}_metrics.json      # METEOR/CIDEr scores
#   {output_dir}/hico_action_v3_{timestamp}.log               # Full log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-qwen-v3-referring}"
BASE_MODEL_PATH="${3:-checkpoints/Qwen3-VL-8B-Instruct}"
HICO_ROOT="../data/hico_20160224_det"
BENCHMARK_ANN="groma_data/benchmarks/hico_action_referring_test.json"
IMAGES_DIR="${HICO_ROOT}/images/test2015"
OUTPUT_DIR="${4:-results-groma-qwen/hico_action_referring_v3}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_v3_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/hico_action_v3_${TIMESTAMP}.json"

echo "========================================================================"
echo "[GROMA-QWEN V3] HICO-DET Action Referring Evaluation"
echo "========================================================================"
echo "GPU:          $GPU_ID"
echo "Model:        $MODEL_PATH"
echo "Base Model:   $BASE_MODEL_PATH"
echo "Annotation:   $BENCHMARK_ANN"
echo "Images:       $IMAGES_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Log file:     $LOG_FILE"
echo ""
echo "Features:"
echo "  ✓ Auto-detects V2 vs V3 models"
echo "  ✓ V3: Uses interaction token (union of person + object boxes)"
echo "  ✓ V2: Standard 2-box format"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$BENCHMARK_ANN" ]; then
    echo "ERROR: Benchmark annotation file not found at $BENCHMARK_ANN"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    exit 1
fi

# Detect model type
if [ -f "$MODEL_PATH/config.json" ]; then
    MODEL_TYPE=$(grep -o '"model_type"[[:space:]]*:[[:space:]]*"[^"]*"' "$MODEL_PATH/config.json" | sed 's/.*"\([^"]*\)"$/\1/')
    if [ "$MODEL_TYPE" = "groma_qwen_interaction" ]; then
        echo "✓ Detected V3 model (interaction token enabled)"
    else
        echo "✓ Detected V2 model (standard 2-box format)"
    fi
fi

# Count test images
NUM_IMAGES=$(ls -1 "$IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

# Parse optional flags
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
WANDB_PROJECT_FLAG=""
WANDB_RUN_NAME_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES triplets"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ W&B logging enabled"
    
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_PROJECT_FLAG="--wandb-project $WANDB_PROJECT"
        echo "  Project: $WANDB_PROJECT"
    fi
    
    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_RUN_NAME_FLAG="--wandb-run-name $WANDB_RUN_NAME"
        echo "  Run name: $WANDB_RUN_NAME"
    fi
fi

echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation using V3 script (handles both V2 and V3)
python groma/eval/eval_action_referring_groma_qwen_v3.py \
    --model-name "$MODEL_PATH" \
    --base-model-name "$BASE_MODEL_PATH" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$BENCHMARK_ANN" \
    --pred-file "$PRED_FILE" \
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
    echo "  Metrics:      ${PRED_FILE//.json/_metrics.json}"
    echo "  Log:          $LOG_FILE"
    echo ""
    if [ ! -z "$VERBOSE" ]; then
        echo "Verbose outputs:"
        echo "  Per-triplet:  ${PRED_FILE//.json/_per_triplet.json}"
        echo "  Per-action:   ${PRED_FILE//.json/_per_action.json}"
        echo ""
    fi
    echo "View metrics:"
    echo "  cat ${PRED_FILE//.json/_metrics.json} | jq '.'"
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Evaluation failed! Check log: $LOG_FILE"
    exit 1
fi

