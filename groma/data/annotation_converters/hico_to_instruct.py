"""
HICO-DET to Groma Grounding Instruction Converter

Converts HICO-DET annotations to Groma's grounding instruction format
for HOI detection fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# Import HICO categories
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))


# Load HICO categories by directly parsing the file to avoid spacy dependency
def load_hico_categories():
    """Load HICO categories from hico_categories.py file."""
    hico_cat_file = Path(__file__).parent.parent.parent / 'eval' / 'hoi_eval' / 'hico_categories.py'

    # Read and execute to get HICO_ACTIONS and HICO_OBJECTS
    with open(hico_cat_file, 'r') as f:
        content = f.read()

    # Create a namespace to execute the code
    namespace = {}
    exec(content, namespace)

    return namespace['HICO_ACTIONS'], namespace['HICO_OBJECTS']


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
    Normalize HICO action names to natural language.

    Converts underscores to spaces and adjusts verb forms for natural language.
    """
    # Convert underscores to spaces
    action_name = action_name.replace('_', ' ')

    # Special cases for verb forms (convert to -ing form for natural descriptions)
    VERB_FORMS = {
        'adjust': 'adjusting',
        'assemble': 'assembling',
        'block': 'blocking',
        'blow': 'blowing',
        'board': 'boarding',
        'break': 'breaking',
        'brush with': 'brushing with',
        'buy': 'buying',
        'carry': 'carrying',
        'catch': 'catching',
        'chase': 'chasing',
        'check': 'checking',
        'clean': 'cleaning',
        'control': 'controlling',
        'cook': 'cooking',
        'cut': 'cutting',
        'cut with': 'cutting with',
        'direct': 'directing',
        'drag': 'dragging',
        'dribble': 'dribbling',
        'drink with': 'drinking with',
        'drive': 'driving',
        'dry': 'drying',
        'eat': 'eating',
        'eat at': 'eating at',
        'exit': 'exiting',
        'feed': 'feeding',
        'fill': 'filling',
        'flip': 'flipping',
        'flush': 'flushing',
        'fly': 'flying',
        'greet': 'greeting',
        'grind': 'grinding',
        'groom': 'grooming',
        'herd': 'herding',
        'hit': 'hitting',
        'hold': 'holding',
        'hop on': 'hopping on',
        'hose': 'hosing',
        'hug': 'hugging',
        'hunt': 'hunting',
        'inspect': 'inspecting',
        'install': 'installing',
        'jump': 'jumping',
        'kick': 'kicking',
        'kiss': 'kissing',
        'lasso': 'lassoing',
        'launch': 'launching',
        'lick': 'licking',
        'lie on': 'lying on',
        'lift': 'lifting',
        'light': 'lighting',
        'load': 'loading',
        'lose': 'losing',
        'make': 'making',
        'milk': 'milking',
        'move': 'moving',
        'no interaction': 'no interaction',
        'open': 'opening',
        'operate': 'operating',
        'pack': 'packing',
        'paint': 'painting',
        'park': 'parking',
        'pay': 'paying',
        'peel': 'peeling',
        'pet': 'petting',
        'pick': 'picking',
        'pick up': 'picking up',
        'point': 'pointing',
        'pour': 'pouring',
        'pull': 'pulling',
        'push': 'pushing',
        'race': 'racing',
        'read': 'reading',
        'release': 'releasing',
        'repair': 'repairing',
        'ride': 'riding',
        'row': 'rowing',
        'run': 'running',
        'sail': 'sailing',
        'scratch': 'scratching',
        'serve': 'serving',
        'set': 'setting',
        'shear': 'shearing',
        'sign': 'signing',
        'sip': 'sipping',
        'sit at': 'sitting at',
        'sit on': 'sitting on',
        'slide': 'sliding',
        'smell': 'smelling',
        'spin': 'spinning',
        'squeeze': 'squeezing',
        'stab': 'stabbing',
        'stand on': 'standing on',
        'stand under': 'standing under',
        'stick': 'sticking',
        'stir': 'stirring',
        'stop at': 'stopping at',
        'straddle': 'straddling',
        'swing': 'swinging',
        'tag': 'tagging',
        'talk on': 'talking on',
        'teach': 'teaching',
        'text on': 'texting on',
        'throw': 'throwing',
        'tie': 'tying',
        'toast': 'toasting',
        'train': 'training',
        'turn': 'turning',
        'type on': 'typing on',
        'walk': 'walking',
        'wash': 'washing',
        'watch': 'watching',
        'wave': 'waving',
        'wear': 'wearing',
        'wield': 'wielding',
        'zip': 'zipping',
    }

    return VERB_FORMS.get(action_name, action_name)


def normalize_object_name(object_name: str) -> str:
    """Normalize HICO object names to natural language."""
    # Convert underscores to spaces
    return object_name.replace('_', ' ')


def convert_hico_to_grounding_instructions(
    anno_file: str,
    output_file: str,
    num_variations: int = 3,
    split: str = 'trainval',
    max_images: Optional[int] = None
) -> None:
    """
    Convert HICO-DET annotations to Groma grounding instruction format.

    Args:
        anno_file: Path to HICO annotation JSON (trainval_hico_ann.json)
        output_file: Path to output instruction JSON
        num_variations: Number of template variations per image
        split: 'trainval' or 'test'
        max_images: Maximum number of images to process (None for all)
    """
    print(f"\n{'='*80}")
    print(f"HICO-DET to Groma Grounding Instruction Converter")
    print(f"{'='*80}\n")

    print(f"Loading HICO annotations from: {anno_file}")
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} images")

    # Load HICO categories
    print(f"Loading HICO categories...")
    HICO_ACTIONS, HICO_OBJECTS = load_hico_categories()

    # Create mappings
    action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}

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
        if not ann.get('hoi_annotation'):
            skipped += 1
            skipped_reasons['no_hoi'] += 1
            continue

        # Generate multiple variations per image
        for var_idx in range(num_variations):
            instruction = convert_single_image_hico(
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


def convert_single_image_hico(
    ann: Dict,
    action_id2name: Dict,
    object_id2name: Dict,
    template_idx: int = 0
) -> Optional[Dict]:
    """
    Convert single HICO annotation to instruction format.

    Args:
        ann: HICO annotation dict for one image
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
    box_annotations = ann['annotations']
    hoi_annotations = ann['hoi_annotation']

    for hoi in hoi_annotations:
        try:
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']

            # Get bounding boxes (keep in [x, y, w, h] format)
            person_box = box_annotations[subject_id]['bbox']
            object_box = box_annotations[object_id]['bbox']

            # Get action ID (CRITICAL: hoi['category_id'] is 1-indexed, subtract 1)
            action_id = hoi['category_id'] - 1

            # Get object category ID (this directly matches HICO_OBJECTS id field)
            object_category_id = box_annotations[object_id]['category_id']

            # Check if action ID is valid
            if action_id not in action_id2name:
                # print(f"WARNING: Invalid action_id {action_id} (original: {hoi['category_id']}) in {ann['file_name']}")
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
    """Command-line interface for HICO converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert HICO-DET annotations to Groma grounding instruction format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--anno-file',
        type=str,
        required=True,
        help='Path to HICO annotation file (trainval_hico_ann.json or test_hico_ann.json)'
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
    convert_hico_to_grounding_instructions(
        anno_file=args.anno_file,
        output_file=args.output_file,
        num_variations=args.num_variations,
        max_images=args.max_images
    )

    return 0


if __name__ == '__main__':
    exit(main())
