# Stage 3: Instruction Finetuning - HICO-DET + SWIG-HOI (Combined Grounding + Action Referring)
#
# This config uses combined HOI datasets with interleaved grounding and action referring tasks.
# Uses the CURRENT format (matching the separate files that were used previously).
#
# Key Characteristics:
# - Grounding: Multi-pair per action-object (e.g., all "person riding motorcycle" pairs)
# - Action Referring: Action + object output (e.g., "riding motorcycle")
# - Interleaved: Both tasks for same image are in the same file
# - Format: Same as separate files (action_object_id for grounding, refer_box for referring)
#
# Expected Performance:
# - This format should restore model performance by:
#   1. Maintaining the CURRENT format the model was recently trained on
#   2. Re-introducing task interleaving for better multi-task learning
#   3. Preserving the action+object output format
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!

datasets = [
    # ========================================================================
    # HICO-DET: Combined Grounding + Action Referring (CURRENT FORMAT)
    # ========================================================================
    {
        'type': 'hico_ground',  # Uses same dataset class as grounding
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/hico_combined_ground_refer_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    },

    # ========================================================================
    # SWIG-HOI: Combined Grounding + Action Referring (CURRENT FORMAT)
    # ========================================================================
    {
        'type': 'swig_ground',  # Uses same dataset class as grounding
        'ann_file': '/Users/shaunliew/Documents/Groma/groma_data/benchmarks/swig_combined_ground_refer_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images_512',
        'conv_temp': 'llava'
    },
]

# Dataset Statistics:
#
# HICO-DET Combined:
# - Total: 162,445 samples
# - Grounding: 69,404 samples (multi-pair detection)
# - Action Referring: 93,041 samples (action + object recognition)
#
# SWIG-HOI Combined:
# - Total: 120,312 samples
# - Grounding: 59,519 samples (multi-pair detection)
# - Action Referring: 60,793 samples (action + object recognition, includes 9,938 person-to-person)
#
# Total: 282,757 samples
#
# Key Advantages of Combined Format:
# 1. Task Interleaving: Model sees grounding and referring for same image consecutively
# 2. Consistent Format: Uses exact same format as separate files (proven to work)
# 3. Natural Task Switching: Better multi-task learning through interleaved examples
# 4. Maintains Current Format: No prompt/response template changes
#
# Comparison with Separate Files (Previous Training):
# - Separate: 306,262 samples (slightly more due to dataset variations)
# - Combined: 282,757 samples (same images, same format, interleaved)
# - Expected: Similar or BETTER performance due to task interleaving
#
# Training Time Estimate (1 epoch):
# - 8 GPUs: ~5-6 hours
# - 4 GPUs: ~10-12 hours
# - 1 GPU: ~20-24 hours
#
# Stage 3 Key Differences from Stage 2:
# - LLM is UNFROZEN and trainable
# - Lower learning rate (2e-5 vs 1e-4)
# - Only 1 epoch instead of 2
# - Uses FSDP for memory efficiency
