#!/usr/bin/env python3
"""
HICO-DET to COCO Detection Format Converter
Converts HICO-DET annotations to class-agnostic COCO format for detection training.

Usage:
    python groma/data/annotation_converters/hico_to_coco_det.py [HICO_ROOT_PATH]

Example:
    python groma/data/annotation_converters/hico_to_coco_det.py ~/Documents/Intent-Identification-Detection/hico_20160224_det
"""

import json
import os
import sys
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable


def convert_hico_to_coco_det(hico_ann_file, output_file):
    """
    Convert HICO-DET annotation to COCO detection format (class-agnostic).

    HICO format:
    [
        {
            "file_name": "HICO_train2015_00000001.jpg",
            "img_id": 1,
            "width": 640,
            "height": 480,
            "annotations": [
                {"bbox": [x1, y1, x2, y2], "category_id": 1}
            ],
            "hoi_annotation": [...]  # Ignored for detection training
        }
    ]

    COCO format:
    {
        "images": [...],
        "annotations": [...],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}]
    }
    """
    print(f"Loading HICO-DET annotations from: {hico_ann_file}")
    with open(hico_ann_file, 'r') as f:
        hico_data = json.load(f)

    print(f"Total images: {len(hico_data)}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "HICO-DET converted to COCO format for class-agnostic detection",
            "version": "1.0",
            "year": 2024,
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "object",
                "supercategory": "object"
            }
        ]
    }

    annotation_id = 1
    skipped_invalid_boxes = 0

    for img_data in tqdm(hico_data, desc="Converting annotations"):
        # Add image info
        image_info = {
            "id": img_data["img_id"],
            "file_name": img_data["file_name"],
            "width": img_data["width"],
            "height": img_data["height"]
        }
        coco_data["images"].append(image_info)

        # Convert each bbox annotation
        for ann in img_data["annotations"]:
            bbox_xyxy = ann["bbox"]  # [x1, y1, x2, y2]

            # Convert to COCO format [x, y, width, height]
            x1, y1, x2, y2 = bbox_xyxy
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1

            # Skip invalid boxes
            if width <= 0 or height <= 0:
                skipped_invalid_boxes += 1
                continue

            area = width * height

            coco_ann = {
                "id": annotation_id,
                "image_id": img_data["img_id"],
                "category_id": 1,  # Class-agnostic: all objects are category 1
                "bbox": [x, y, width, height],
                "area": area,
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_ann)
            annotation_id += 1

    # Save to output file
    print(f"\nSaving COCO format annotations to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

    print(f"\n✅ Conversion complete!")
    print(f"   - Total images: {len(coco_data['images'])}")
    print(f"   - Total annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])} (class-agnostic)")
    print(f"   - Skipped invalid boxes: {skipped_invalid_boxes}")

    return coco_data


def main():
    """Convert both train and test splits"""

    # Get HICO root from command line or use default
    if len(sys.argv) > 1:
        hico_root = sys.argv[1]
    else:
        hico_root = os.path.expanduser(
            "~/Documents/Intent-Identification-Detection/hico_20160224_det"
        )

    print(f"\nUsing HICO-DET root: {hico_root}\n")

    # Verify paths exist
    trainval_ann = f"{hico_root}/annotations/trainval_hico_ann.json"
    test_ann = f"{hico_root}/annotations/test_hico_ann.json"

    if not os.path.exists(trainval_ann):
        print(f"ERROR: Training annotation file not found: {trainval_ann}")
        sys.exit(1)

    if not os.path.exists(test_ann):
        print(f"ERROR: Test annotation file not found: {test_ann}")
        sys.exit(1)

    # Convert training set
    print("=" * 60)
    print("Converting HICO-DET Training Set")
    print("=" * 60)
    convert_hico_to_coco_det(
        hico_ann_file=trainval_ann,
        output_file=f"{hico_root}/annotations/hico_det_train_coco.json"
    )

    # Convert test set
    print("\n" + "=" * 60)
    print("Converting HICO-DET Test Set")
    print("=" * 60)
    convert_hico_to_coco_det(
        hico_ann_file=test_ann,
        output_file=f"{hico_root}/annotations/hico_det_test_coco.json"
    )

    print("\n" + "=" * 60)
    print("All conversions complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {hico_root}/annotations/hico_det_train_coco.json")
    print(f"  - {hico_root}/annotations/hico_det_test_coco.json")


if __name__ == "__main__":
    main()
