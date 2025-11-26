# Stage 3: Instruction Finetuning - Referring-Only (Qwen3VL-Native Instructions)
#
# This config uses ONLY referring samples with new explicit instructions for Qwen3VL.
# - Excludes all grounding samples to avoid task confusion
# - Uses new Qwen3VL-native instruction format with explicit task description
# - Designed for Stage 3 fine-tuning starting from Stage 2 checkpoint
#
# New Instruction Format:
#   System: "You are an expert at understanding human-object interactions..."
#   User: "Action Recognition Task: The first highlighted region ... contains a PERSON..."
#   Assistant: "<action phrase>" (e.g., "riding bicycle", "sitting on bench")
#
# ⚠️ IMPORTANT: Update image paths to match your server setup!

datasets = [
    # HICO-DET Referring Only (Action Recognition)
    {
        'type': 'groma_qwen_instruct',
        'ann_file': 'groma_data/benchmarks/hico_referring_only_qwen3vl.json',
        'img_prefix': '../data/hico_20160224_det/images/train2015',
    },

    # SWIG-HOI Referring Only (Action Recognition)
    {
        'type': 'groma_qwen_instruct',
        'ann_file': 'groma_data/benchmarks/swig_referring_only_qwen3vl.json',
        'img_prefix': '../data/swig_hoi/images_512',
    },
]

# Dataset Statistics:
# - HICO-DET Referring: 93,041 samples
# - SWIG-HOI Referring: 60,793 samples
# - Total: 153,834 samples (referring only, no grounding)
#
# Comparison with combined dataset:
# - Original combined: 282,757 samples (grounding + referring)
# - This config: 153,834 samples (referring only)
# - Reduction: 45.6% of original size

