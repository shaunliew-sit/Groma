# Stage 3: Instruction Finetuning - HICO-DET + SWIG-HOI (Grounding + Action Referring)
#
# This config uses HOI datasets for both grounding and action referring tasks
# - Grounding: Detect entities and describe interactions (format: "person riding motorcycle")
# - Action Referring: Given person+object boxes, output action+object (format: "riding motorcycle")
#
# Key Changes:
# - Action referring now outputs "action + object" instead of just "action"
# - Includes person-to-person interactions in SWIG (e.g., "talking person")
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!

datasets = [
    # ========================================================================
    # HICO-DET: Grounding Task (Detection + Description)
    # ========================================================================
    {
        'type': 'hico_ground',
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/hico_ground_train.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    },

    # ========================================================================
    # HICO-DET: Action Referring Task (Action Recognition with Action+Object Output)
    # ========================================================================
    {
        'type': 'hico_action_referring',
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/hico_action_referring_train.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    },

    # ========================================================================
    # SWIG-HOI: Grounding Task (Detection + Description)
    # ========================================================================
    {
        'type': 'swig_ground',
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/swig_ground_train.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images_512',
        'conv_temp': 'llava'
    },

    # ========================================================================
    # SWIG-HOI: Action Referring Task (Action Recognition with Action+Object Output)
    # Includes person-to-person interactions (e.g., "talking person")
    # ========================================================================
    {
        'type': 'swig_action_referring',
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/swig_action_referring_train.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images_512',
        'conv_temp': 'llava'
    },
]

# Dataset Statistics:
#
# HICO-DET:
# - Grounding: 69,404 samples (detection + description)
# - Action Referring: 93,041 samples (action + object recognition)
# - Subtotal: 162,445 samples
#
# SWIG-HOI:
# - Grounding: 83,024 samples (detection + description)
# - Action Referring: 60,793 samples (action + object recognition, 16.3% person-to-person)
# - Subtotal: 143,817 samples
#
# Total: 306,262 samples (both grounding and referring tasks)
#
# Key Differences from Previous Version:
# - Action referring now outputs "action + object" (e.g., "riding motorcycle" not just "riding")
# - Supports person-to-person interactions in SWIG (e.g., "talking person", "hugging person")
# - Uses new benchmark files from groma_data/benchmarks/
# - More comprehensive training with both detection and recognition tasks
#
# Training Time Estimate (1 epoch):
# - 8 GPUs: ~5-7 hours (increased due to more samples)
# - 4 GPUs: ~10-14 hours
# - 1 GPU: ~20-28 hours
#
# Stage 3 Key Differences from Stage 2:
# - LLM is now UNFROZEN and trainable
# - Lower learning rate (2e-5 vs 1e-4)
# - Only 1 epoch instead of 2
# - Uses FSDP for memory efficiency
