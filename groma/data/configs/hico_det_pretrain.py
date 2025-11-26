# HICO-DET Detection Fine-tuning Configuration
# Class-agnostic object detection training on HICO-DET dataset
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!
#
# This config is used by groma/train/train_det.py for Stage 1 detection pretraining.
# All object categories are treated as a single class "object" for class-agnostic detection.

datasets = [
    {
        'type': 'coco_box',
        'ann_file': '../data/hico_20160224_det/annotations/hico_det_train_coco.json',
        'img_prefix': '../data/hico_20160224_det/images/train2015',
    }
]

# Optional: Combine with other detection datasets for better generalization
# Uncomment to use multiple datasets:
#
# datasets = [
#     # COCO detection
#     {
#         'type': 'coco_box',
#         'ann_file': '/path/to/coco/annotations/instances_train2017.json',
#         'img_prefix': '/path/to/coco/train2017',
#     },
#     # HICO-DET detection
#     {
#         'type': 'coco_box',
#         'ann_file': '/path/to/hico_20160224_det/annotations/hico_det_train_coco.json',
#         'img_prefix': '/path/to/hico_20160224_det/images/train2015',
#     },
#     # Objects365
#     {
#         'type': 'obj365_box',
#         'ann_file': '/path/to/Objects365/annotations/zhiyuan_objv2_train.json',
#         'img_prefix': '/path/to/Objects365/train',
#     },
# ]
