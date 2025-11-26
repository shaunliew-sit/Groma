#!/usr/bin/env python3
"""
Prepare HICO-DET Action Referring dataset in COCO caption format.

Input:  test_hico_ann.json (HICO-DET format)
Output: hico_action_referring_test.json (COCO caption format)

Each HOI triplet becomes a "caption task":
- Image ID = unique triplet ID (e.g., "img1_person0_object1_action87")
- Caption = ground truth action (e.g., "sit on")
- Metadata = person bbox, object bbox, categories

Usage:
    python scripts/prepare_hico_action_referring_data.py \
        --input-file /path/to/test_hico_ann.json \
        --output-file groma_data/hico_action_referring_test.json
"""

import json
import argparse
import os
import sys
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import HICO action and object names
from groma.eval.hoi_eval.hico_categories import HICO_ACTIONS, HICO_OBJECTS


def convert_hico_to_action_referring(input_file, output_file):
    """Convert HICO-DET to action referring COCO format"""

    print(f"Loading HICO-DET annotations from: {input_file}")
    with open(input_file, 'r') as f:
        hico_data = json.load(f)

    # Build action and object ID mappings
    action_id_to_name = {a['id']: a['name'] for a in HICO_ACTIONS}
    object_id_to_name = {o['id']: o['name'] for o in HICO_OBJECTS}

    # COCO format structure
    coco_data = {
        "info": {
            "description": "HICO-DET Action Referring Task",
            "version": "1.0",
            "date_created": "2024",
            "contributor": "Groma Action Referring Evaluation"
        },
        "images": [],
        "annotations": [],
        "licenses": []
    }

    # Track statistics
    stats = defaultdict(int)
    ann_id = 0

    for item in hico_data:
        img_id = item['img_id']
        file_name = item['file_name']
        width = item['width']
        height = item['height']

        # Process each HOI triplet in this image
        for hoi in item.get('hoi_annotation', []):
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']
            action_id = hoi['category_id']

            # Get action name
            action_name = action_id_to_name.get(action_id, "unknown")

            # Skip "no_interaction" cases
            if action_name == "no_interaction":
                stats['skipped_no_interaction'] += 1
                continue

            # Get bboxes (format: [x, y, w, h] in absolute pixels)
            subject_bbox = item['annotations'][subject_id]['bbox']
            object_bbox = item['annotations'][object_id]['bbox']

            # Get category IDs
            subject_cat_id = item['annotations'][subject_id]['category_id']
            object_cat_id = item['annotations'][object_id]['category_id']

            # Get category names
            subject_cat_name = object_id_to_name.get(subject_cat_id, "person")
            object_cat_name = object_id_to_name.get(object_cat_id, "object")

            # Create unique triplet ID
            # Format: {original_image_id}_{person_idx}_{object_idx}_{action_id}
            unique_triplet_id = f"{img_id}_{subject_id}_{object_id}_{action_id}"

            # Add "image" entry (one per triplet - each triplet is treated as separate "image")
            coco_data["images"].append({
                "id": unique_triplet_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "original_image_id": img_id,
                "subject_bbox": subject_bbox,  # [x, y, w, h] absolute
                "object_bbox": object_bbox,    # [x, y, w, h] absolute
                "subject_category": subject_cat_name,
                "object_category": object_cat_name,
                "action_id": action_id
            })

            # Add annotation (ground truth action as "caption")
            # Convert underscores to spaces: "sit_on" → "sit on"
            action_caption = action_name.replace('_', ' ')

            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": unique_triplet_id,
                "caption": action_caption,
                "action_id": action_id,
                "action_name_original": action_name  # Keep original for reference
            })

            stats[f'action_{action_name}'] += 1
            stats['total_triplets'] += 1
            ann_id += 1

    # Save output
    print(f"\nSaving COCO-format dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Preparation Complete")
    print(f"{'='*70}")
    print(f"Total unique triplets (images): {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Skipped (no_interaction): {stats['skipped_no_interaction']}")

    # Get unique original images count
    unique_imgs = len(set([img['original_image_id'] for img in coco_data['images']]))
    print(f"Unique original images: {unique_imgs}")

    print(f"\nTop-15 most frequent actions:")
    action_counts = [(k.replace('action_', ''), v)
                     for k, v in stats.items() if k.startswith('action_')]
    action_counts.sort(key=lambda x: x[1], reverse=True)

    for action, count in action_counts[:15]:
        action_display = action.replace('_', ' ')
        print(f"  {action_display:20s}: {count:4d} samples")

    print(f"{'='*70}\n")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Ready for evaluation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HICO-DET to action referring COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/prepare_hico_action_referring_data.py \\
        --input-file /Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/test_hico_ann.json \\
        --output-file groma_data/hico_action_referring_test.json

    # With custom output directory
    python scripts/prepare_hico_action_referring_data.py \\
        --input-file ../data/hico/test_hico_ann.json \\
        --output-file results/custom_location/hico_action_test.json
        """
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to test_hico_ann.json (HICO-DET format)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="groma_data/hico_action_referring_test.json",
        help="Output path for COCO-format JSON (default: groma_data/hico_action_referring_test.json)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    convert_hico_to_action_referring(args.input_file, args.output_file)
