# [QWEN3VL ORIGINAL] Dataset Configuration for LoRA Fine-tuning
#
# This config uses existing Qwen3VL-native datasets for training
# the original Qwen3-VL model (without Groma modifications).
#
# Task Types (controlled by TASK_TYPE env var):
# - referring: Action recognition (native bbox tokens)
# - grounding: Object localization (JSON format)
# - both: Combined training
#
# Usage:
#   TASK_TYPE=referring python -m groma.train.train_qwen3vl_original_lora ...
#   TASK_TYPE=grounding python -m groma.train.train_qwen3vl_original_lora ...
#   TASK_TYPE=both python -m groma.train.train_qwen3vl_original_lora ...

import os

# Get task type from environment variable
TASK_TYPE = os.environ.get('TASK_TYPE', 'both')
print(f"[Qwen3VL Original Config] Task type: {TASK_TYPE}")

# Dataset configuration based on task type
if TASK_TYPE == 'referring':
    # Referring-only datasets (action recognition)
    # Uses native Qwen3VL bbox tokens: <|box_start|>, <|box_end|>
    datasets = [
        # HICO-DET Referring Only (93,041 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/hico_referring_qwen3vl_native.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
        },
        # SWIG-HOI Referring Only (60,793 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/swig_referring_qwen3vl_native.json',
            'img_prefix': '../data/swig_hoi/images_512',
        },
    ]
    print(f"[Qwen3VL Original Config] Using referring-only datasets (153,834 samples)")
    
elif TASK_TYPE == 'grounding':
    # Grounding-only datasets (JSON format)
    # Uses JSON format aligned with Qwen3-VL: [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]
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
    print(f"[Qwen3VL Original Config] Using grounding-only datasets (128,923 samples)")
    
else:  # 'both' or default
    # Combined datasets (both referring and grounding)
    # Referring uses native bbox tokens, grounding uses JSON format
    datasets = [
        # HICO-DET Combined (162,445 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/hico_combined_qwen3vl_json.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
        },
        # SWIG-HOI Combined (120,312 samples)
        {
            'type': 'groma_qwen_instruct',
            'ann_file': 'groma_data/benchmarks/swig_combined_qwen3vl_json.json',
            'img_prefix': '../data/swig_hoi/images_512',
        },
    ]
    print(f"[Qwen3VL Original Config] Using combined datasets (282,757 samples)")

# ============================================================================
# Dataset Statistics:
# ============================================================================
#
# Referring-only (native bbox tokens):
#   Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
#   - HICO-DET: 93,041 samples
#   - SWIG-HOI: 60,793 samples
#   - Total: 153,834 samples
#
# Grounding-only (JSON format):
#   Format: [{"bbox_2d": [x1, y1, x2, y2], "label": "person/object"}]
#   - HICO-DET: 69,404 samples
#   - SWIG-HOI: 59,519 samples
#   - Total: 128,923 samples
#
# Combined:
#   - HICO-DET: 162,445 samples
#   - SWIG-HOI: 120,312 samples
#   - Total: 282,757 samples
#
# ============================================================================

