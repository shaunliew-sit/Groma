#!/usr/bin/env python3
"""
SWIG-HOI to COCO Detection Format Converter
Converts SWIG-HOI annotations to class-agnostic COCO format for detection training.

Usage:
    python groma/data/annotation_converters/swig_to_coco_det.py [SWIG_ROOT_PATH]

Example:
    python groma/data/annotation_converters/swig_to_coco_det.py ~/Documents/Intent-Identification-Detection/swig_hoi
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


def convert_swig_to_coco_det(swig_ann_file, output_file):
    """
    Convert SWIG-HOI annotation to COCO detection format (class-agnostic).

    SWIG format:
    [
        {
            "file_name": "talking_90.jpg",
            "img_id": 0,
            "width": 769,
            "height": 512,
            "box_annotations": [
                {"bbox": [x1, y1, x2, y2], "category_id": 0, "aux_category_id": []}
            ],
            "hoi_annotations": [...]  # Ignored for detection training
        }
    ]

    COCO format:
    {
        "images": [...],
        "annotations": [...],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}]
    }
    """
    print(f"Loading SWIG-HOI annotations from: {swig_ann_file}")
    with open(swig_ann_file, 'r') as f:
        swig_data = json.load(f)

    print(f"Total images: {len(swig_data)}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "SWIG-HOI converted to COCO format for class-agnostic detection",
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
    total_boxes = 0

    for img_data in tqdm(swig_data, desc="Converting annotations"):
        # Add image info
        image_info = {
            "id": img_data["img_id"],
            "file_name": img_data["file_name"],
            "width": img_data["width"],
            "height": img_data["height"]
        }
        coco_data["images"].append(image_info)

        # Convert each bbox annotation
        # SWIG uses 'box_annotations' (note the 's')
        for box_ann in img_data.get("box_annotations", []):
            total_boxes += 1
            bbox_xyxy = box_ann["bbox"]  # [x1, y1, x2, y2]

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

            # Clip to image boundaries
            x = max(0, min(x, img_data["width"]))
            y = max(0, min(y, img_data["height"]))
            width = min(width, img_data["width"] - x)
            height = min(height, img_data["height"] - y)

            # Skip if clipped box is invalid
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
    print(f"   - Total boxes processed: {total_boxes}")
    print(f"   - Valid annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])} (class-agnostic)")
    print(f"   - Skipped invalid boxes: {skipped_invalid_boxes}")
    print(f"   - Avg boxes per image: {len(coco_data['annotations']) / len(coco_data['images']):.2f}")

    return coco_data


def main():
    """Convert both train and test splits"""

    # Get SWIG root from command line or use default
    if len(sys.argv) > 1:
        swig_root = sys.argv[1]
    else:
        swig_root = os.path.expanduser(
            "~/Documents/Intent-Identification-Detection/swig_hoi"
        )

    print(f"\nUsing SWIG-HOI root: {swig_root}\n")

    # Verify paths exist
    trainval_ann = f"{swig_root}/annotations/swig_trainval_1000.json"
    test_ann = f"{swig_root}/annotations/swig_test_1000.json"

    # Also support separate train/val files if they exist
    train_ann = f"{swig_root}/annotations/swig_train_1000.json"
    val_ann = f"{swig_root}/annotations/swig_val_1000.json"

    # Determine which files to convert
    files_to_convert = []

    if os.path.exists(trainval_ann):
        files_to_convert.append({
            'input': trainval_ann,
            'output': f"{swig_root}/annotations/swig_det_trainval_coco.json",
            'name': 'TrainVal'
        })
    elif os.path.exists(train_ann):
        files_to_convert.append({
            'input': train_ann,
            'output': f"{swig_root}/annotations/swig_det_train_coco.json",
            'name': 'Train'
        })

    if os.path.exists(test_ann):
        files_to_convert.append({
            'input': test_ann,
            'output': f"{swig_root}/annotations/swig_det_test_coco.json",
            'name': 'Test'
        })

    if os.path.exists(val_ann):
        files_to_convert.append({
            'input': val_ann,
            'output': f"{swig_root}/annotations/swig_det_val_coco.json",
            'name': 'Val'
        })

    if not files_to_convert:
        print("ERROR: No SWIG annotation files found!")
        print(f"Expected files in {swig_root}/annotations/:")
        print("  - swig_trainval_1000.json or swig_train_1000.json")
        print("  - swig_test_1000.json")
        sys.exit(1)

    # Convert each file
    for file_info in files_to_convert:
        print("=" * 60)
        print(f"Converting SWIG-HOI {file_info['name']} Set")
        print("=" * 60)
        convert_swig_to_coco_det(
            swig_ann_file=file_info['input'],
            output_file=file_info['output']
        )
        print()

    print("=" * 60)
    print("All conversions complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    for file_info in files_to_convert:
        print(f"  - {file_info['output']}")


if __name__ == "__main__":
    main()
