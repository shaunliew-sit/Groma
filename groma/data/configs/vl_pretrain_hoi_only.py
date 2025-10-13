# Stage 2: VL Alignment Pretraining - HICO-DET + SWIG-HOI Only
#
# This config uses only HOI datasets for VL alignment
# - Focused on human-object interaction understanding
# - Smaller dataset, faster training
# - No need to download additional data
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!

datasets = [
    # HICO-DET: Human-Object Interaction grounding instructions
    {
        'type': 'hico_hoi_instruct',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/groma_instructions/hico_grounding_instruct_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'default'
    },

    # SWIG-HOI: Situation With Interacting Groundings grounding instructions
    {
        'type': 'swig_hoi_instruct',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/annotations/groma_instructions/swig_grounding_instruct_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images_512',
        'conv_temp': 'default'
    },
]

# Dataset Statistics:
# - HICO-DET: 112,899 instruction samples (37,633 images × 3 variations)
# - SWIG-HOI: 163,803 instruction samples (54,601 images × 3 variations)
# - Total: 276,702 samples
#
# Training Time Estimate (2 epochs):
# - 8 GPUs: ~8-12 hours (vs ~12-18 hours for standard datasets)
# - 4 GPUs: ~16-24 hours
# - 1 GPU: ~32-48 hours
#
# Note: This is smaller than standard Groma's ~1.5M samples
# Consider training for 3-4 epochs instead of 2 to compensate:
#   --num_train_epochs 4
