# Stage 3: Instruction Finetuning - HICO-DET + SWIG-HOI Only
#
# This config uses only HOI datasets for instruction finetuning
# - Same datasets as Stage 2, but now training the LLM
# - LLM learns to follow instructions using the aligned vision-language bridge
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!

datasets = [
    # HICO-DET: Human-Object Interaction grounding instructions
    {
        'type': 'hico_hoi_instruct',
        'ann_file': '../data/hico_20160224_det/annotations/groma_instructions/hico_grounding_instruct_train.json',
        'img_prefix': '../data/hico_20160224_det/images/train2015',
        'conv_temp': 'default'
    },

    # SWIG-HOI: Situation With Interacting Groundings grounding instructions
    {
        'type': 'swig_hoi_instruct',
        'ann_file': '../data/swig_hoi/annotations/groma_instructions/swig_grounding_instruct_train.json',
        'img_prefix': '../data/swig_hoi/images_512',
        'conv_temp': 'default'
    },
]

# Dataset Statistics:
# - HICO-DET: 112,899 instruction samples (37,633 images × 3 variations)
# - SWIG-HOI: 163,803 instruction samples (54,601 images × 3 variations)
# - Total: 276,702 samples
#
# Training Time Estimate (1 epoch):
# - 8 GPUs: ~4-6 hours (vs ~6-8 hours for standard datasets)
# - 4 GPUs: ~8-12 hours
# - 1 GPU: ~16-24 hours
#
# Stage 3 Key Differences from Stage 2:
# - LLM is now UNFROZEN and trainable
# - Lower learning rate (2e-5 vs 1e-4)
# - Only 1 epoch instead of 2
# - Uses FSDP for memory efficiency
