#!/usr/bin/env python3
"""
SWIG-HOI to Groma Grounding Instruction Converter

Converts SWIG-HOI annotations to Groma's grounding instruction format
for HOI detection fine-tuning.

Adapted from hico_to_instruct.py for SWIG-HOI format.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# Import SWIG categories
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))


# Load SWIG categories by directly parsing the file
def load_swig_categories():
    """Load SWIG categories from swig_v1_categories.py file."""
    swig_cat_file = Path(__file__).parent.parent.parent / 'eval' / 'hoi_eval' / 'swig_v1_categories.py'

    # Read and execute to get SWIG_ACTIONS and SWIG_CATEGORIES
    with open(swig_cat_file, 'r') as f:
        content = f.read()

    # Create a namespace to execute the code
    namespace = {}
    exec(content, namespace)

    return namespace['SWIG_ACTIONS'], namespace['SWIG_CATEGORIES']


# Query templates for instruction diversity
HOI_GROUNDING_TEMPLATES = [
    "Describe what each person is doing with objects individually. Focus on actions only.",
    "What actions are the people performing with objects?",
    "Describe the human-object interactions in this image.",
    "What are people doing with the objects around them?",
    "Identify all human-object interactions in this scene.",
    "Describe how objects are being used by people.",
    "Explain what each person is doing in relation to objects.",
    "What is happening in this image? Focus on interactions.",
]


def normalize_action_name(action_name: str) -> str:
    """
    Normalize SWIG action names to natural language.

    SWIG actions are already in -ing form (e.g., 'adjusting', 'blowing'),
    so less normalization needed compared to HICO.
    """
    # Convert underscores to spaces
    action_name = action_name.replace('_', ' ')

    # SWIG action names are already mostly in good form
    # Just ensure lowercase
    return action_name.lower()


def normalize_object_name(object_name: str) -> str:
    """Normalize SWIG object names to natural language."""
    # Convert underscores to spaces
    return object_name.replace('_', ' ')


def convert_swig_to_grounding_instructions(
    anno_file: str,
    output_file: str,
    num_variations: int = 3,
    split: str = 'trainval',
    max_images: Optional[int] = None
) -> None:
    """
    Convert SWIG-HOI annotations to Groma grounding instruction format.

    Args:
        anno_file: Path to SWIG annotation JSON (swig_trainval_1000.json)
        output_file: Path to output instruction JSON
        num_variations: Number of template variations per image
        split: 'trainval', 'train', 'test', or 'val'
        max_images: Maximum number of images to process (None for all)
    """
    print(f"\n{'='*80}")
    print(f"SWIG-HOI to Groma Grounding Instruction Converter")
    print(f"{'='*80}\n")

    print(f"Loading SWIG annotations from: {anno_file}")
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} images")

    # Load SWIG categories
    print(f"Loading SWIG categories...")
    SWIG_ACTIONS, SWIG_CATEGORIES = load_swig_categories()

    # Create mappings
    action_id2name = {x["id"]: x["name"] for x in SWIG_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in SWIG_CATEGORIES}

    print(f"Loaded {len(action_id2name)} actions and {len(object_id2name)} objects")

    # Limit images if specified
    if max_images is not None:
        annotations = annotations[:max_images]
        print(f"Processing subset of {len(annotations)} images")

    instructions = []
    skipped = 0
    skipped_reasons = defaultdict(int)

    for idx, ann in enumerate(annotations):
        # Skip images without HOI annotations
        if not ann.get('hoi_annotations'):
            skipped += 1
            skipped_reasons['no_hoi'] += 1
            continue

        # Generate multiple variations per image
        for var_idx in range(num_variations):
            instruction = convert_single_image_swig(
                ann, action_id2name, object_id2name,
                template_idx=var_idx % len(HOI_GROUNDING_TEMPLATES)
            )
            if instruction:
                instructions.append(instruction)
            else:
                if var_idx == 0:  # Only count once per image
                    skipped += 1
                    skipped_reasons['conversion_failed'] += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(annotations)} images, generated {len(instructions)} instructions...")

    # Save to JSON
    print(f"\nSaving {len(instructions)} instruction examples to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(instructions, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Conversion Complete!")
    print(f"{'='*80}")
    print(f"✓ Generated {len(instructions)} instruction examples from {len(annotations)} images")
    print(f"  - Valid images: {len(annotations) - skipped}")
    print(f"  - Skipped images: {skipped}")
    if skipped_reasons:
        print(f"  - Skip reasons:")
        for reason, count in skipped_reasons.items():
            print(f"    * {reason}: {count}")
    print(f"  - Variations per image: {num_variations}")
    print(f"  - Output file: {output_file}")
    print(f"{'='*80}\n")


def convert_single_image_swig(
    ann: Dict,
    action_id2name: Dict,
    object_id2name: Dict,
    template_idx: int = 0
) -> Optional[Dict]:
    """
    Convert single SWIG annotation to instruction format.

    Args:
        ann: SWIG annotation dict for one image
        action_id2name: Mapping from action ID to action name
        object_id2name: Mapping from object ID to object name
        template_idx: Index of query template to use

    Returns:
        Instruction dict in Groma format, or None if conversion fails
    """

    boxes = []
    box_inds = []
    hoi_triplets = []

    # Extract all HOI triplets
    # SWIG uses 'box_annotations' and 'hoi_annotations' (note the 's')
    box_annotations = ann['box_annotations']
    hoi_annotations = ann['hoi_annotations']

    for hoi in hoi_annotations:
        try:
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']

            # Get bounding boxes (SWIG format: [x1, y1, x2, y2])
            person_box = box_annotations[subject_id]['bbox']
            object_box = box_annotations[object_id]['bbox']

            # Get action ID (SWIG uses 'action_id' directly)
            action_id = hoi['action_id']

            # Get object category ID
            object_category_id = box_annotations[object_id]['category_id']

            # Check if action ID is valid
            if action_id not in action_id2name:
                # print(f"WARNING: Invalid action_id {action_id} in {ann['file_name']}")
                continue

            # Check if object ID is valid
            if object_category_id not in object_id2name:
                # print(f"WARNING: Invalid object_id {object_category_id} in {ann['file_name']}")
                continue

            # Get names
            action_name = action_id2name[action_id]
            object_name = object_id2name[object_category_id]

            # Normalize names
            action_name = normalize_action_name(action_name)
            object_name = normalize_object_name(object_name)

            hoi_triplets.append({
                'person_box': person_box,
                'object_box': object_box,
                'action': action_name,
                'object': object_name
            })

        except IndexError as e:
            # print(f"ERROR: Index error in {ann['file_name']}: {e}")
            continue
        except KeyError as e:
            # print(f"ERROR: Key error in {ann['file_name']}: {e}")
            continue
        except Exception as e:
            # print(f"ERROR: Failed to process HOI in {ann['file_name']}: {e}")
            continue

    if not hoi_triplets:
        return None

    # Build boxes array and response text
    response_parts = []

    for i, triplet in enumerate(hoi_triplets):
        # Add person and object boxes
        person_idx = len(boxes)
        object_idx = len(boxes) + 1

        boxes.append(triplet['person_box'])
        boxes.append(triplet['object_box'])
        box_inds.extend([person_idx, object_idx])

        # Build grounded response sentence
        # Use different sentence structures for variety
        # NOTE: Using <ground_box> tokens (not <r#>) - box_inds determines mapping
        if i == 0:
            # First HOI: "The person is..."
            sentence = f"The <p> person </p> <roi> <ground_box> </roi> is <p> {triplet['action']} </p> the <p> {triplet['object']} </p> <roi> <ground_box> </roi>."
        elif i == 1:
            # Second HOI: "The person is also..."
            sentence = f" The <p> person </p> <roi> <ground_box> </roi> is also <p> {triplet['action']} </p> a <p> {triplet['object']} </p> <roi> <ground_box> </roi>."
        elif i == 2:
            # Third HOI: "Additionally, the person is..."
            sentence = f" Additionally, the <p> person </p> <roi> <ground_box> </roi> is <p> {triplet['action']} </p> a <p> {triplet['object']} </p> <roi> <ground_box> </roi>."
        else:
            # Fourth+ HOI: "Another person is..."
            sentence = f" Another <p> person </p> <roi> <ground_box> </roi> is <p> {triplet['action']} </p> a <p> {triplet['object']} </p> <roi> <ground_box> </roi>."

        response_parts.append(sentence)

    response_text = "".join(response_parts).strip()

    # Select query template
    query = HOI_GROUNDING_TEMPLATES[template_idx]

    # Build instruction in Groma format
    # Match the exact format from groma_instruct.json
    instruction = {
        "file_name": ann['file_name'],
        "width": ann.get('width', 640),
        "height": ann.get('height', 480),
        "boxes": boxes,
        "conversation": [
            {
                "from": "human",
                "value": query,
                "box_inds": None  # Use None (null in JSON) for human queries
            },
            {
                "from": "gpt",
                "value": response_text,
                "box_inds": box_inds
            }
        ]
    }

    return instruction


def main():
    """Command-line interface for SWIG converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert SWIG-HOI annotations to Groma grounding instruction format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--anno-file',
        type=str,
        required=True,
        help='Path to SWIG annotation file (swig_trainval_1000.json or swig_test_1000.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to output instruction JSON file'
    )
    parser.add_argument(
        '--num-variations',
        type=int,
        default=3,
        help='Number of template variations per image'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (None for all)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.anno_file).exists():
        print(f"ERROR: Annotation file not found: {args.anno_file}")
        return 1

    # Run conversion
    convert_swig_to_grounding_instructions(
        anno_file=args.anno_file,
        output_file=args.output_file,
        num_variations=args.num_variations,
        max_images=args.max_images
    )

    return 0


if __name__ == '__main__':
    exit(main())
