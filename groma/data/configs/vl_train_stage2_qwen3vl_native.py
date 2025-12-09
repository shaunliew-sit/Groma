# [GROMA-QWEN V2] Stage 2 Training Configuration - JSON Format for Grounding
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# Training Strategy:
# ==================
# Stage 2 Only (frozen Qwen3VL):
# - Freeze: Qwen3VL LLM + Qwen3VL Vision Encoder
# - Train: ROI Align module + Image-to-Text Bridge
#
# Architecture:
# =============
# - Referring: Uses native Qwen3VL bbox tokens (<|box_start|>, <|box_end|>)
# - Grounding: Uses JSON format ({"bbox_2d": [x1, y1, x2, y2], "label": "..."})
# - No custom tokens, no DINOv2 encoder
# - ROI Align on Qwen3VL vision features
#
# Dataset Selection:
# ==================
# Set TASK_TYPE environment variable to control dataset:
# - 'referring': Train on referring task only (action recognition)
# - 'grounding': Train on grounding task only (JSON format, object localization)
# - 'both': Train on both tasks (default)
#
# Example:
#   TASK_TYPE=referring python -m groma.train.train_qwen ...
#   TASK_TYPE=grounding python -m groma.train.train_qwen ...
#   TASK_TYPE=both python -m groma.train.train_qwen ...
#
# ⚠️ IMPORTANT: Update image paths to match your server setup!

import os

# Get task type from environment variable (default: both)
TASK_TYPE = os.environ.get('TASK_TYPE', 'both')

print(f"[Config] Task type: {TASK_TYPE}")

# Dataset configuration based on task type
if TASK_TYPE == 'referring':
    # Referring-only datasets (action recognition)
    datasets = [
        # HICO-DET Referring Only
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/hico_referring_qwen3vl_native.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
        },
        # SWIG-HOI Referring Only
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/swig_referring_qwen3vl_native.json',
            'img_prefix': '../data/swig_hoi/images_512',
        },
    ]
    print(f"[Config] Using referring-only datasets (153,834 samples)")
    
elif TASK_TYPE == 'grounding':
    # Grounding-only datasets (JSON format, object localization)
    # Uses dedicated grounding-only files with JSON format aligned with Qwen3-VL
    datasets = [
        # HICO-DET grounding-only (69,404 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/hico_grounding_only_json.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
        },
        # SWIG-HOI grounding-only (59,519 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/swig_grounding_only_json.json',
            'img_prefix': '../data/swig_hoi/images_512',
        },
    ]
    print(f"[Config] Using grounding-only datasets with JSON format (128,923 samples)")
    
else:  # 'both' or default
    # Combined datasets (both referring and grounding with JSON format)
    # Grounding uses JSON format aligned with Qwen3-VL
    datasets = [
        # HICO-DET Combined (JSON format for grounding)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/hico_combined_qwen3vl_json.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
        },
        # SWIG-HOI Combined (JSON format for grounding)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/swig_combined_qwen3vl_json.json',
            'img_prefix': '../data/swig_hoi/images_512',
        },
    ]
    print(f"[Config] Using combined datasets with JSON grounding format (282,757 samples)")

# Dataset Statistics:
# ===================
# Referring-only (native bbox tokens):
#   - HICO-DET: 93,041 samples
#   - SWIG-HOI: 60,793 samples
#   - Total: 153,834 samples
#
# Grounding-only (JSON format - aligned with Qwen3-VL):
#   - HICO-DET: 69,404 samples (hico_grounding_only_json.json)
#   - SWIG-HOI: 59,519 samples (swig_grounding_only_json.json)
#   - Total: 128,923 samples
#   - Format: [{"bbox_2d": [x1, y1, x2, y2], "label": "person/object_category"}, ...]
#
# Combined (both):
#   - HICO-DET: 162,445 samples
#   - SWIG-HOI: 120,312 samples
#   - Total: 282,757 samples

# Training Configuration Notes:
# =============================
# Recommended training parameters for Stage 2:
#
# --freeze_llm True              # Freeze Qwen3VL LLM
# --freeze_perceiver True        # Freeze vision encoder (not applicable in V2)
# --freeze_vl_bridge False       # Train Image-to-Text Bridge
# --per_device_train_batch_size 4
# --gradient_accumulation_steps 4
# --num_train_epochs 4
# --learning_rate 2e-4
# --warmup_ratio 0.03
# --lr_scheduler_type cosine
#
# Example command:
# ================
# TASK_TYPE=referring deepspeed --num_gpus=8 \
#     -m groma.train.train_qwen \
#     --llm checkpoints/Qwen3-VL-8B-Instruct \
#     --output_dir checkpoints/groma-qwen-v2-stage2-referring \
#     --dataset_config groma/data/configs/vl_train_stage2_qwen3vl_native.py \
#     --freeze_llm True \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 4 \
#     --learning_rate 2e-4 \
#     --warmup_ratio 0.03 \
#     --bf16 True \
#     --deepspeed scripts/zero2.json

