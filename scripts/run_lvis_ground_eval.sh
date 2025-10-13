#!/bin/bash
################################################################################
# LVIS-Ground Evaluation Script
# Replicates Table 3 results from the Groma paper
#
# Usage:
#   bash scripts/run_lvis_ground_eval.sh [GPU] [MODEL] [ANN_FILE] [IMAGES_DIR] [OUTPUT_DIR]
#
# Examples:
#   bash scripts/run_lvis_ground_eval.sh 0                    # Use GPU 0 with defaults
#   bash scripts/run_lvis_ground_eval.sh 1                    # Use GPU 1 with defaults
#   bash scripts/run_lvis_ground_eval.sh 0,1                  # Use GPU 0 and 1
#   bash scripts/run_lvis_ground_eval.sh 2 /path/to/model     # Use GPU 2 with custom model
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-7b-finetune}"
LVIS_ANN="${3:-data_images/lvis_ground/lvis_test.json}"
IMAGES_DIR="${4:-data_images/lvis_ground/images}"
OUTPUT_DIR="${5:-results/lvis_ground}"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/lvis_evaluation_${TIMESTAMP}.log"
RESULT_FILE="$OUTPUT_DIR/lvis_predictions_${TIMESTAMP}.json"

# GPU availability check (optional, shows info but doesn't fail)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "LVIS-Ground Evaluation"
echo "========================================================================"
echo "GPU:         $GPU_ID"
echo "Model:       $MODEL_PATH"
echo "Annotation:  $LVIS_ANN"
echo "Images:      $IMAGES_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download: git clone https://huggingface.co/FoundationVision/groma-7b-finetune $MODEL_PATH"
    exit 1
fi

if [ ! -f "$LVIS_ANN" ]; then
    echo "ERROR: Annotation file not found at $LVIS_ANN"
    echo "Please run: python scripts/setup_lvis_ground.py"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    echo "Please run: python scripts/setup_lvis_ground.py"
    exit 1
fi

echo "Starting evaluation..."
echo ""

# Run evaluation
python groma/eval/eval_lvis.py \
    --model-name "$MODEL_PATH" \
    --img-prefix "$IMAGES_DIR" \
    --ann-file "$LVIS_ANN" \
    --result-file "$RESULT_FILE" \
    --box_score_thres 0.15 \
    --batch_size_per_gpu 1 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Evaluation complete!"
echo "========================================================================"
echo "Results saved to: $LOG_FILE"
echo ""
echo "Expected results from paper (Table 3):"
echo "  AR:      28.8"
echo "  AR@0.5:  37.9"
echo "  AR@0.75: 30.3"
echo "  AR@s:     8.7"
echo "  AR@m:    35.6"
echo "  AR@l:    64.3"
echo "========================================================================"
