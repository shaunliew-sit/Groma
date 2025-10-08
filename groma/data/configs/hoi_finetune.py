# HOI Fine-tuning Configuration
# Pure HICO-DET grounding instruction fine-tuning
# Format: Matches Groma Instruct format exactly (v2 with <ground_box> tokens)

datasets = [
    # HICO-DET HOI Grounding Instructions
    {
        'type': 'hico_hoi_instruct',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/groma_instructions/hico_grounding_instruct_train_v2.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    }
]
