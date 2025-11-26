#!/usr/bin/env python3
"""
Prepare Action Referring dataset in COCO caption format from Groma instruction format.

Input:  latest_hico_grounding_referring_test.json OR latest_swig_grounding_referring_test.json
Output: action_referring_test.json (COCO caption format)

Each HOI triplet (from referring conversation) becomes a "caption task":
- Image ID = unique triplet ID
- Caption = ground truth action (e.g., "sitting on", "holding")
- Metadata = person bbox, object bbox

Usage:
    # HICO
    python scripts/prepare_action_referring_from_groma_instruct.py \
        --input-file /path/to/latest_hico_grounding_referring_test.json \
        --output-file groma_data/hico_action_referring_test.json

    # SWIG
    python scripts/prepare_action_referring_from_groma_instruct.py \
        --input-file /path/to/latest_swig_grounding_referring_test.json \
        --output-file groma_data/swig_action_referring_test.json
"""

import json
import argparse
import os
import re
from collections import defaultdict


def convert_groma_instruct_to_action_referring(input_file, output_file):
    """
    Convert Groma grounding+referring instruction format to action referring COCO format.

    Only extracts the REFERRING tasks (ignores grounding tasks).
    """

    print(f"Loading Groma instruction data from: {input_file}")
    with open(input_file, 'r') as f:
        instruct_data = json.load(f)

    # COCO format structure
    coco_data = {
        "info": {
            "description": "Action Referring Task (from Groma Instructions)",
            "version": "2.0",
            "date_created": "2024"
        },
        "images": [],
        "annotations": [],
        "licenses": []
    }

    # Track statistics
    stats = defaultdict(int)
    ann_id = 0
    triplet_idx = 0

    for item_idx, item in enumerate(instruct_data):
        file_name = item['file_name']
        width = item['width']
        height = item['height']
        boxes = item['boxes']
        conversation = item['conversation']

        # Only process REFERRING tasks
        # Referring conversation has:
        #   human: "Describe only the action that <roi> <refer_box> </roi> <refer_feat> is doing to..."
        #   gpt: "sitting on" (or other action)
        is_referring = False
        for conv in conversation:
            if conv['from'] == 'human' and '<refer_box>' in conv['value']:
                is_referring = True
                break

        if not is_referring:
            stats['skipped_grounding'] += 1
            continue

        # Extract action from GPT response
        action = None
        box_inds = None
        for conv in conversation:
            if conv['from'] == 'human' and '<refer_box>' in conv['value']:
                box_inds = conv.get('box_inds', [])
            elif conv['from'] == 'gpt':
                action = conv['value'].strip()

        if not action or not box_inds or len(box_inds) < 2:
            stats['skipped_invalid'] += 1
            continue

        # Skip "no interaction" actions
        if 'no interaction' in action.lower():
            stats['skipped_no_interaction'] += 1
            continue

        # Get person and object boxes
        person_idx = box_inds[0]
        object_idx = box_inds[1]

        if person_idx >= len(boxes) or object_idx >= len(boxes):
            stats['skipped_invalid_indices'] += 1
            continue

        person_bbox_xyxy = boxes[person_idx]
        object_bbox_xyxy = boxes[object_idx]

        # Convert [x1, y1, x2, y2] to [x, y, w, h]
        person_bbox = [
            person_bbox_xyxy[0],
            person_bbox_xyxy[1],
            person_bbox_xyxy[2] - person_bbox_xyxy[0],
            person_bbox_xyxy[3] - person_bbox_xyxy[1]
        ]
        object_bbox = [
            object_bbox_xyxy[0],
            object_bbox_xyxy[1],
            object_bbox_xyxy[2] - object_bbox_xyxy[0],
            object_bbox_xyxy[3] - object_bbox_xyxy[1]
        ]

        # Extract original image ID from filename
        # HICO format: HICO_test2015_00000001.jpg -> 1
        # SWIG format: various formats
        img_id_match = re.search(r'(\d+)', file_name)
        if img_id_match:
            original_img_id = int(img_id_match.group(1))
        else:
            original_img_id = item_idx

        # Create unique triplet ID
        unique_triplet_id = f"{original_img_id}_{person_idx}_{object_idx}_{triplet_idx}"
        triplet_idx += 1

        # Add "image" entry (one per triplet)
        coco_data["images"].append({
            "id": unique_triplet_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "original_image_id": original_img_id,
            "subject_bbox": person_bbox,  # [x, y, w, h]
            "object_bbox": object_bbox,   # [x, y, w, h]
            "subject_category": "person",
            "object_category": "object"  # Could extract from grounding if needed
        })

        # Add annotation (ground truth action as "caption")
        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": unique_triplet_id,
            "caption": action  # Already in correct form (e.g., "sitting on")
        })

        stats[f'action_{action}'] += 1
        stats['total_triplets'] += 1
        ann_id += 1

        if (ann_id) % 1000 == 0:
            print(f"  Processed {ann_id} triplets...")

    # Save output
    print(f"\nSaving COCO-format dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Preparation Complete")
    print(f"{'='*70}")
    print(f"Total triplets (images): {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Skipped (grounding tasks): {stats['skipped_grounding']}")
    print(f"Skipped (no_interaction): {stats['skipped_no_interaction']}")
    print(f"Skipped (invalid): {stats['skipped_invalid']}")
    print(f"Skipped (invalid indices): {stats['skipped_invalid_indices']}")

    # Get unique original images count
    unique_imgs = len(set([img['original_image_id'] for img in coco_data['images']]))
    print(f"Unique original images: {unique_imgs}")

    print(f"\nTop-15 most frequent actions:")
    action_counts = [(k.replace('action_', ''), v)
                     for k, v in stats.items() if k.startswith('action_')]
    action_counts.sort(key=lambda x: x[1], reverse=True)

    for action, count in action_counts[:15]:
        print(f"  {action:20s}: {count:4d} samples")

    print(f"{'='*70}\n")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Ready for evaluation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Groma instruction format to action referring COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # HICO
    python scripts/prepare_action_referring_from_groma_instruct.py \\
        --input-file /Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/annotations/groma_instructions/latest_hico_grounding_referring_test.json \\
        --output-file groma_data/hico_action_referring_test.json

    # SWIG
    python scripts/prepare_action_referring_from_groma_instruct.py \\
        --input-file /Users/shaunliew/Documents/Intent-Identification-Detection/swig_hoi/annotations/groma_instructions/latest_swig_grounding_referring_test.json \\
        --output-file groma_data/swig_action_referring_test.json
        """
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to latest_*_grounding_referring_test.json (Groma instruction format)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for COCO-format action referring JSON"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        exit(1)

    convert_groma_instruct_to_action_referring(args.input_file, args.output_file)
