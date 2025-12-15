# [GROMA-QWEN V3] Stage 2 Training Configuration - Interaction Token
# Part of: Groma Qwen3VL with Interaction Token Architecture
#
# Training Strategy:
# ==================
# Stage 2 Only (frozen Qwen3VL):
# - Freeze: Qwen3VL LLM + Qwen3VL Vision Encoder
# - Train: ROI Align module + Image-to-Text Bridge
#
# Architecture (V3):
# ==================
# - Adds interaction token (union of person + object boxes)
# - 3 region features instead of 2
# - Uses GromaInstructQwenInteraction dataset class
# - Referring: Uses native Qwen3VL bbox tokens (<|box_start|>, <|box_end|>)
# - Grounding: Uses JSON format ({"bbox_2d": [x1, y1, x2, y2], "label": "..."})
#
# Dataset Selection:
# ==================
# Set TASK_TYPE environment variable to control dataset:
# - 'referring': Train on referring task only (action recognition) - RECOMMENDED for V3
# - 'grounding': Train on grounding task only (JSON format)
# - 'both': Train on both tasks
#
# Example:
#   TASK_TYPE=referring python -m groma.train.train_qwen_interaction ...
#   TASK_TYPE=grounding python -m groma.train.train_qwen_interaction ...
#   TASK_TYPE=both python -m groma.train.train_qwen_interaction ...
#
# ⚠️ IMPORTANT: Update image paths to match your server setup!

import os

# Get task type from environment variable (default: referring for V3)
TASK_TYPE = os.environ.get('TASK_TYPE', 'referring')

print(f"[V3 Config] Task type: {TASK_TYPE}")
print(f"[V3 Config] Using interaction token (union box)")

# Dataset configuration based on task type
if TASK_TYPE == 'referring':
    # Referring-only datasets (action recognition) with interaction token
    datasets = [
        # HICO-DET Referring Only
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/hico_referring_qwen3vl_native.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
            'use_interaction': True,  # V3: Enable interaction token
        },
        # SWIG-HOI Referring Only
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/swig_referring_qwen3vl_native.json',
            'img_prefix': '../data/swig_hoi/images_512',
            'use_interaction': True,  # V3: Enable interaction token
        },
    ]
    print(f"[V3 Config] Using referring-only datasets with interaction token (153,834 samples)")
    
elif TASK_TYPE == 'grounding':
    # Grounding-only datasets (JSON format)
    # Note: Grounding doesn't use interaction token (no input boxes)
    datasets = [
        # HICO-DET grounding-only
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/hico_grounding_only_json.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
            'use_interaction': False,  # Grounding: no input boxes, so no interaction
        },
        # SWIG-HOI grounding-only
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/swig_grounding_only_json.json',
            'img_prefix': '../data/swig_hoi/images_512',
            'use_interaction': False,  # Grounding: no input boxes, so no interaction
        },
    ]
    print(f"[V3 Config] Using grounding-only datasets (128,923 samples)")
    print(f"[V3 Config] Note: Interaction token disabled for grounding (no input boxes)")
    
else:  # 'both' or default
    # Combined datasets (both referring with interaction and grounding)
    datasets = [
        # HICO-DET Combined
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/hico_combined_qwen3vl_json.json',
            'img_prefix': '../data/hico_20160224_det/images/train2015',
            'use_interaction': True,  # V3: Enable interaction token for referring
        },
        # SWIG-HOI Combined
        {
            'type': 'groma_qwen_interaction_instruct',
            'ann_file': 'groma_data/benchmarks/swig_combined_qwen3vl_json.json',
            'img_prefix': '../data/swig_hoi/images_512',
            'use_interaction': True,  # V3: Enable interaction token for referring
        },
    ]
    print(f"[V3 Config] Using combined datasets with interaction token (282,757 samples)")
    print(f"[V3 Config] Note: Interaction token applies to referring samples only")

# Dataset Statistics:
# ===================
# Referring-only (with interaction token):
#   - HICO-DET: 93,041 samples
#   - SWIG-HOI: 60,793 samples
#   - Total: 153,834 samples
#   - Each sample has 3 boxes: person, object, interaction (union)
#
# Grounding-only (no interaction token):
#   - HICO-DET: 69,404 samples
#   - SWIG-HOI: 59,519 samples
#   - Total: 128,923 samples
#
# Combined:
#   - Referring samples get interaction token
#   - Grounding samples don't (no input boxes)

# Training Configuration Notes:
# =============================
# Recommended training parameters for V3 Stage 2:
#
# --freeze_llm True              # Freeze Qwen3VL LLM
# --freeze_vl_bridge False       # Train Image-to-Text Bridge
# --use_interaction_token True   # Enable V3 interaction token
# --per_device_train_batch_size 4
# --gradient_accumulation_steps 4
# --num_train_epochs 4
# --learning_rate 2e-4
# --warmup_ratio 0.03
# --lr_scheduler_type cosine
#
# Example command (referring only from V2 checkpoint):
# ====================================================
# TASK_TYPE=referring CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     -m groma.train.train_qwen_interaction \
#     --model_name_or_path checkpoints/groma-qwen-v2-stage2-referring-new \
#     --output_dir checkpoints/groma-qwen-v3-referring \
#     --dataset_config groma/data/configs/vl_train_stage2_qwen3vl_interaction.py \
#     --freeze_llm True \
#     --use_interaction_token True \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 4 \
#     --learning_rate 2e-4 \
#     --warmup_ratio 0.03 \
#     --bf16 True

