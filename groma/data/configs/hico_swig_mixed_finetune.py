# HICO-DET + SWIG-HOI Mixed Fine-tuning Configuration
# Combined HOI grounding instruction fine-tuning on both datasets
# Format: Matches Groma Instruct format exactly (v2 with <ground_box> tokens)

datasets = [
    # HICO-DET HOI Grounding Instructions
    {
        'type': 'hico_hoi_instruct',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/groma_instructions/hico_grounding_instruct_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    },
    # SWIG-HOI Grounding Instructions
    {
        'type': 'swig_hoi_instruct',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/annotations/groma_instructions/swig_grounding_instruct_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images',
        'conv_temp': 'llava'
    }
]

# Training on combined dataset:
# - HICO-DET: 37,633 images → ~112,899 instruction samples
# - SWIG-HOI: 54,601 images → ~163,803 instruction samples
# - TOTAL: 92,234 images → ~276,702 instruction samples
#
# Benefits:
# - Better generalization (real photos + movie frames)
# - More training data (2.4x larger than HICO alone)
# - Single model works on both HICO and SWIG benchmarks
