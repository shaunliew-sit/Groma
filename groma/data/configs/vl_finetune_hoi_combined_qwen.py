# [GROMA-QWEN] Stage 3: Instruction Finetuning - HICO-DET + SWIG-HOI Combined (Qwen-Only Grounding)
# Part of: Groma Qwen3VL Referring Task Implementation
#
# This config uses the COMBINED HOI datasets for Instruction Finetuning
# - Uses the same datasets as Stage 2 for consistency
# - Uses 'groma_qwen_instruct' to generate text coordinates instead of <ground_box> tokens
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!

datasets = [
    # HICO-DET Combined: Grounding + Action Referring
    {
        'type': 'groma_qwen_instruct',
        'ann_file': 'groma_data/benchmarks/hico_qwen_v3_qwen3vl.json',
        'img_prefix': '../data/hico_20160224_det/images/train2015',
    },

    # SWIG-HOI Combined: Grounding + Action Referring
    {
        'type': 'groma_qwen_instruct',
        'ann_file': 'groma_data/benchmarks/swig_qwen_v3_qwen3vl.json',
        'img_prefix': '../data/swig_hoi/images_512',
    },
]

# Dataset Statistics:
# - HICO-DET Combined: 162,445 samples
# - SWIG-HOI Combined: 120,312 samples
# - Total: 282,757 samples
