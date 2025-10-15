#!/bin/bash
################################################################################
# HICO-Grounding Evaluation Script
# Tests HOI interaction grounding capability using instruction format
#
# Usage:
#   bash scripts/run_hoi_grounding_eval.sh [GPU] [MODEL] [MAX_IMAGES] [USE_GROUNDING_PREFIX]
#
# Examples:
#   bash scripts/run_hoi_grounding_eval.sh 0                          # Use GPU 0, all images, WITH [grounding]
#   bash scripts/run_hoi_grounding_eval.sh 1                          # Use GPU 1, all images, WITH [grounding]
#   bash scripts/run_hoi_grounding_eval.sh 0 /path/to/model           # Custom model, WITH [grounding]
#   bash scripts/run_hoi_grounding_eval.sh 0 checkpoints/groma 100    # First 100 images, WITH [grounding]
#   bash scripts/run_hoi_grounding_eval.sh 0 checkpoints/groma 100 0  # WITHOUT [grounding] prefix
#   bash scripts/run_hoi_grounding_eval.sh 0 checkpoints/groma 0 1    # All images, WITH [grounding]
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_PATH="${2:-checkpoints/groma-7b-finetune}"
MAX_IMAGES="${3:-0}"  # 0 = all images
USE_GROUNDING_PREFIX="${4:-1}"  # 1 = use [grounding] prefix, 0 = no prefix

# HICO-DET paths (can be overridden with environment variables)
HICO_ROOT="${HICO_ROOT:-../data/hico_20160224_det}"
INSTRUCTION_FILE="${INSTRUCTION_FILE:-${HICO_ROOT}/annotations/groma_instructions/hico_grounding_instruct_test.json}"
IMAGES_DIR="${IMAGES_DIR:-${HICO_ROOT}/images/test2015}"
OUTPUT_DIR="results/hoi_grounding"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Extract model name from path for subfolder organization
# e.g., "checkpoints/groma-7b-finetune" -> "groma-7b-finetune"
MODEL_NAME=$(basename "$MODEL_PATH")

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create model-specific subdirectory with timestamp
# Structure: results/hoi_grounding/{model_name}/{timestamp}/
RUN_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME/$TIMESTAMP"
mkdir -p "$RUN_OUTPUT_DIR"

# File naming with grounding suffix
GROUNDING_SUFFIX=$([ $USE_GROUNDING_PREFIX -eq 1 ] && echo "with_grounding" || echo "no_grounding")
LOG_FILE="$RUN_OUTPUT_DIR/hoi_grounding_${GROUNDING_SUFFIX}.log"
RESULT_FILE="$RUN_OUTPUT_DIR/hoi_grounding_${GROUNDING_SUFFIX}.json"

# GPU availability check (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-Grounding Evaluation"
echo "========================================================================"
echo "GPU:              $GPU_ID"
echo "Model:            $MODEL_PATH"
echo "Model name:       $MODEL_NAME"
echo "Instruction file: $INSTRUCTION_FILE"
echo "Images:           $IMAGES_DIR"
echo "Output directory: $RUN_OUTPUT_DIR"
echo "Log file:         $LOG_FILE"
echo "Result file:      $RESULT_FILE"
echo "Max images:       $([ $MAX_IMAGES -eq 0 ] && echo 'All (9,546)' || echo $MAX_IMAGES)"
echo "Grounding prefix: $([ $USE_GROUNDING_PREFIX -eq 1 ] && echo 'YES [grounding] prefix' || echo 'NO prefix')"
echo "Timestamp:        $TIMESTAMP"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download the checkpoint first."
    exit 1
fi

if [ ! -f "$INSTRUCTION_FILE" ]; then
    echo "ERROR: Instruction file not found at $INSTRUCTION_FILE"
    echo ""
    echo "Please generate the test set first:"
    echo "  cd \$HICO_ROOT"
    echo "  python3 /path/to/groma/groma/data/annotation_converters/hico_to_instruct.py --split test"
    echo ""
    echo "Or set INSTRUCTION_FILE environment variable to the correct path."
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    echo "Please download HICO-DET dataset first."
    exit 1
fi

echo "Starting HOI-Grounding evaluation..."
echo ""

# Run evaluation
# Build python command with grounding prefix flag
if [ $USE_GROUNDING_PREFIX -eq 1 ]; then
    GROUNDING_FLAG="--use-grounding-prefix"
else
    GROUNDING_FLAG="--no-grounding-prefix"
fi

python3 groma/eval/eval_hoi_grounding.py \
    --model-name "$MODEL_PATH" \
    --instruction-file "$INSTRUCTION_FILE" \
    --img-prefix "$IMAGES_DIR" \
    --result-file "$RESULT_FILE" \
    --iou-threshold 0.5 \
    --box-score-thres 0.15 \
    --max-images $MAX_IMAGES \
    $GROUNDING_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Evaluation complete!"
echo "========================================================================"
echo "Results saved to: $RESULT_FILE"
echo "Log saved to:     $LOG_FILE"
echo ""
echo "Interpretation Guide:"
echo "  Triplet Accuracy (TA): Percentage of correctly grounded HOI triplets"
echo "    - Both person AND object boxes must match (IoU > 0.5)"
echo "    - Measures end-to-end grounding capability for interactions"
echo ""
echo "  Expected baseline (rough estimates):"
echo "    - Good HOI grounding:  TA > 40%"
echo "    - Moderate grounding:  TA 20-40%"
echo "    - Poor grounding:      TA < 20%"
echo ""
echo "  Compare with LVIS-Ground results:"
echo "    - If TA is low but LVIS AR is high: model struggles with HOI context"
echo "    - If both TA and AR are low: general grounding capability degraded"
echo "========================================================================"
