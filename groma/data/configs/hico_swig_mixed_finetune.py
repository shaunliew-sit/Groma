# HICO-DET + SWIG-HOI Mixed Fine-tuning Configuration (LATEST)
# Combined HOI grounding + referring instruction fine-tuning on both datasets
#
# New features:
# - Dual-task format: Grounding (detection) + Referring (action recognition)
# - Standardized -ing form verbs for all actions
# - Deduplicated boxes: one unique box per entity
# - One action per person-object pair: no redundant descriptions

datasets = [
    # HICO-DET HOI Instructions (LATEST - Grounding + Referring)
    {
        'type': 'hico_hoi_instruct',
        'ann_file': '../data/hico_20160224_det/annotations/groma_instructions/latest_hico_grounding_referring_train.json',
        'img_prefix': '../data/hico_20160224_det/images/train2015',
        'conv_temp': 'llava'
    },
    # SWIG-HOI Instructions (LATEST - Grounding + Referring)
    {
        'type': 'swig_hoi_instruct',
        'ann_file': '../data/swig_hoi/annotations/groma_instructions/latest_swig_grounding_referring_train.json',
        'img_prefix': '../data/swig_hoi/images_512',
        'conv_temp': 'llava'
    }
]

# Training on combined dataset (LATEST):
# - HICO-DET: 130,674 samples (37,633 grounding + 93,041 referring)
# - SWIG-HOI: 102,111 samples (41,318 grounding + 60,793 referring)
# - TOTAL: 232,785 samples (78,951 grounding + 153,834 referring)
#
# Improvements over previous version:
# - No duplicate boxes in training data
# - Actions standardized to -ing form (riding, carrying, sitting on, etc.)
# - Separate referring task for action recognition from bounding boxes
# - One action per unique person-object pair (more efficient training)
#
# Benefits:
# - Better entity-region mapping (no more r10, r10 for different entities)
# - Consistent action verb format across HICO and SWIG
# - Dual-task learning improves both detection and recognition
# - Single model works on both HICO and SWIG benchmarks
