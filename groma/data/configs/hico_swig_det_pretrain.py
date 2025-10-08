# HICO-DET + SWIG-HOI Mixed Detection Pretraining Configuration
# Class-agnostic object detection training on combined HICO-DET and SWIG-HOI datasets
#
# ⚠️ IMPORTANT: Update these paths to match your server setup!
#
# This config is used by groma/train/train_det.py for Stage 1 detection pretraining.
# All object categories are treated as a single class "object" for class-agnostic detection.
#
# Dataset Statistics:
#   HICO-DET Train:     37,633 images, 152,785 boxes (avg 4.06 boxes/image)
#   SWIG-HOI TrainVal:  54,601 images, 134,743 boxes (avg 2.47 boxes/image)
#   Combined Total:     92,234 images, 287,528 boxes (avg 3.12 boxes/image)
#
# Training Benefits:
#   ✅ 2.4x more data than HICO-only
#   ✅ Better generalization across HOI datasets
#   ✅ More robust object detection for downstream tasks

datasets = [
    # HICO-DET detection dataset
    {
        'type': 'coco_box',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/hico_det_train_coco.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/train2015',
    },
    # SWIG-HOI detection dataset
    {
        'type': 'coco_box',
        'ann_file': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/annotations/swig_det_trainval_coco.json',
        'img_prefix': '/Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/images_512',
    }
]

# Optional: Add more detection datasets for even better generalization
# Uncomment to use additional datasets (requires downloading):
#
# datasets = [
#     # HICO-DET
#     {
#         'type': 'coco_box',
#         'ann_file': '/path/to/hico_20160224_det/annotations/hico_det_train_coco.json',
#         'img_prefix': '/path/to/hico_20160224_det/images/train2015',
#     },
#     # SWIG-HOI
#     {
#         'type': 'coco_box',
#         'ann_file': '/path/to/swig_hoi/annotations/swig_det_trainval_coco.json',
#         'img_prefix': '/path/to/swig_hoi/images_512',
#     },
#     # COCO detection (general objects)
#     {
#         'type': 'coco_box',
#         'ann_file': '/path/to/coco/annotations/instances_train2017.json',
#         'img_prefix': '/path/to/coco/train2017',
#     },
#     # Objects365 (large-scale detection)
#     {
#         'type': 'obj365_box',
#         'ann_file': '/path/to/Objects365/annotations/zhiyuan_objv2_train.json',
#         'img_prefix': '/path/to/Objects365/train',
#     },
# ]
