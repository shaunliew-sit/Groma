# [GROMA-QWEN] Stage 2: VL Alignment Pretraining - HICO-DET + SWIG-HOI Combined
# Part of: Groma Qwen3VL Referring Task Implementation
#
# This config uses the COMBINED HOI datasets for VL alignment
# - Ensures consistency with Stage 3 (Finetuning)
# - Uses the same interleaved format
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
