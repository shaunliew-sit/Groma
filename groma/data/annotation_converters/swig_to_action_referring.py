#!/usr/bin/env python3
"""
SWIG-HOI to Action Referring Instruction Converter

Converts SWIG-HOI annotations to Groma's action referring format with action+object output.

Output format: "action object" (e.g., "riding bicycle", "talking person")
Special handling: Person-to-person interactions (e.g., "talking person", "hugging person")

Usage:
    python groma/data/annotation_converters/swig_to_action_referring.py \
        --anno-file /path/to/swig_train_1000.json \
        --output-file groma_data/benchmarks/swig_action_referring_train.json
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import sys


# Import SWIG categories
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_swig_categories():
    """Load SWIG categories from swig_v1_categories.py file."""
    swig_cat_file = Path(__file__).parent.parent.parent / 'eval' / 'hoi_eval' / 'swig_v1_categories.py'

    with open(swig_cat_file, 'r') as f:
        content = f.read()

    namespace = {}
    exec(content, namespace)

    return namespace['SWIG_ACTIONS'], namespace['SWIG_CATEGORIES']


# Query template for action referring task
REFERRING_QUERY_TEMPLATE = "What is <roi> <refer_box> </roi> <refer_feat> doing with <roi> <refer_box> </roi> <refer_feat>? Answer with action and target:"


def normalize_action_to_ing_form(action_name: str) -> str:
    """
    Normalize action names to -ing form for consistent training.
    SWIG actions are already in -ing form, but normalize to be safe.
    """
    action_name = action_name.strip().lower().replace('_', ' ')

    # SWIG actions are already in -ing form
    # Just ensure proper format
    if action_name.endswith('ing'):
        return action_name

    # If not, apply same logic as HICO
    VERB_TO_ING = {
        'adjust': 'adjusting', 'assemble': 'assembling', 'block': 'blocking',
        'blow': 'blowing', 'board': 'boarding', 'break': 'breaking',
        'brush': 'brushing', 'brush with': 'brushing with', 'buy': 'buying',
        'carry': 'carrying', 'catch': 'catching', 'chase': 'chasing',
        'check': 'checking', 'clean': 'cleaning', 'control': 'controlling',
        'cook': 'cooking', 'cut': 'cutting', 'cut with': 'cutting with',
        'direct': 'directing', 'drag': 'dragging', 'dribble': 'dribbling',
        'drink with': 'drinking with', 'drive': 'driving', 'dry': 'drying',
        'eat': 'eating', 'eat at': 'eating at', 'exit': 'exiting',
        'feed': 'feeding', 'fill': 'filling', 'flip': 'flipping',
        'flush': 'flushing', 'fly': 'flying', 'greet': 'greeting',
        'grind': 'grinding', 'groom': 'grooming', 'herd': 'herding',
        'hit': 'hitting', 'hold': 'holding', 'hop on': 'hopping on',
        'hose': 'hosing', 'hug': 'hugging', 'hunt': 'hunting',
        'inspect': 'inspecting', 'install': 'installing', 'jump': 'jumping',
        'kick': 'kicking', 'kiss': 'kissing', 'lasso': 'lassoing',
        'launch': 'launching', 'lick': 'licking', 'lie on': 'lying on',
        'lift': 'lifting', 'light': 'lighting', 'load': 'loading',
        'lose': 'losing', 'make': 'making', 'milk': 'milking',
        'move': 'moving', 'open': 'opening', 'operate': 'operating',
        'pack': 'packing', 'paint': 'painting', 'park': 'parking',
        'pay': 'paying', 'peel': 'peeling', 'pet': 'petting',
        'pick': 'picking', 'pick up': 'picking up', 'point': 'pointing',
        'pour': 'pouring', 'pull': 'pulling', 'push': 'pushing',
        'race': 'racing', 'read': 'reading', 'release': 'releasing',
        'repair': 'repairing', 'ride': 'riding', 'row': 'rowing',
        'run': 'running', 'sail': 'sailing', 'scratch': 'scratching',
        'serve': 'serving', 'set': 'setting', 'shear': 'shearing',
        'sign': 'signing', 'sip': 'sipping', 'sit at': 'sitting at',
        'sit on': 'sitting on', 'slide': 'sliding', 'smell': 'smelling',
        'spin': 'spinning', 'squeeze': 'squeezing', 'stab': 'stabbing',
        'stand on': 'standing on', 'stand under': 'standing under',
        'stick': 'sticking', 'stir': 'stirring', 'stop at': 'stopping at',
        'straddle': 'straddling', 'swing': 'swinging', 'tag': 'tagging',
        'talk on': 'talking on', 'teach': 'teaching', 'text on': 'texting on',
        'throw': 'throwing', 'tie': 'tying', 'toast': 'toasting',
        'train': 'training', 'turn': 'turning', 'type on': 'typing on',
        'walk': 'walking', 'wash': 'washing', 'watch': 'watching',
        'wave': 'waving', 'wear': 'wearing', 'wield': 'wielding',
        'zip': 'zipping', 'no interaction': 'no interaction',
    }

    return VERB_TO_ING.get(action_name, action_name)


def normalize_object_name(object_name: str) -> str:
    """Normalize SWIG object names to natural language."""
    return object_name.replace('_', ' ')


def select_primary_action(actions: List[Dict]) -> Dict:
    """Select most representative action from multiple actions."""

    PHYSICAL_MANIPULATION = {
        'holding', 'carrying', 'picking up', 'cutting', 'opening', 'closing',
        'pushing', 'pulling', 'lifting', 'grabbing', 'throwing', 'catching',
        'adjusting', 'assembling', 'breaking', 'dragging', 'feeding', 'filling',
        'flipping', 'grinding', 'installing', 'launching', 'loading', 'packing',
        'peeling', 'repairing', 'squeezing', 'stirring', 'tying', 'zipping',
        'brushing', 'cutting with', 'brushing with'
    }

    ACTIVE_USE = {
        'riding', 'driving', 'eating', 'drinking', 'reading', 'playing',
        'kicking', 'hitting', 'operating', 'racing', 'sailing', 'flying',
        'typing', 'texting', 'cleaning', 'cooking', 'painting', 'training',
        'washing', 'wielding', 'dribbling', 'rowing', 'spinning', 'swinging',
        'drinking with', 'eating at', 'talking on', 'texting on', 'typing on'
    }

    SPATIAL_RELATION = {
        'sitting on', 'standing on', 'lying on', 'leaning on', 'straddling',
        'standing under', 'sitting at', 'stopping at', 'hopping on'
    }

    SOCIAL_INTERACTION = {
        'talking', 'kissing', 'hugging', 'teaching', 'greeting', 'feeding',
        'herding', 'petting', 'chasing', 'hunting'
    }

    PASSIVE_STATE = {
        'wearing', 'watching', 'looking at', 'smelling', 'inspecting',
        'no interaction'
    }

    def get_priority(action_name):
        if action_name in PHYSICAL_MANIPULATION:
            return 4
        elif action_name in ACTIVE_USE:
            return 3
        elif action_name in SPATIAL_RELATION:
            return 2
        elif action_name in SOCIAL_INTERACTION:
            return 1
        elif action_name in PASSIVE_STATE:
            return 0
        else:
            return 0

    return max(actions, key=lambda a: get_priority(a['action']))


def convert_single_image_swig_action_referring(
    ann: Dict,
    action_id2name: Dict,
    object_id2name: Dict
) -> List[Dict]:
    """
    Convert single SWIG image to action referring format.
    Output: "action object" (e.g., "riding bicycle", "talking person")

    Special handling: Person-to-person interactions (category_id=0 for "person")

    Returns:
        List of instruction dicts (one per HOI triplet)
    """

    instructions = []
    boxes = []
    box_map = {}

    box_annotations = ann['box_annotations']
    hoi_annotations = ann['hoi_annotations']

    # Group HOIs by (person, object) pair
    pair_actions = defaultdict(list)

    for hoi in hoi_annotations:
        try:
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']
            action_id = hoi['action_id']

            person_box = tuple(box_annotations[subject_id]['bbox'])
            object_box = tuple(box_annotations[object_id]['bbox'])
            object_category_id = box_annotations[object_id]['category_id']

            if action_id not in action_id2name or object_category_id not in object_id2name:
                continue

            # Normalize to -ing form (SWIG already has -ing, but be safe)
            action_name = normalize_action_to_ing_form(action_id2name[action_id])
            object_name = normalize_object_name(object_id2name[object_category_id])

            # Filter degenerate boxes (zero area)
            x1, y1, x2, y2 = person_box
            if x2 <= x1 or y2 <= y1:
                continue

            x1, y1, x2, y2 = object_box
            if x2 <= x1 or y2 <= y1:
                continue

            # Note: object_name can be "person" for person-to-person interactions
            # This is expected and will result in outputs like "talking person"

            pair_key = (person_box, object_box)
            pair_actions[pair_key].append({
                'action': action_name,
                'object_name': object_name,
                'person_box': list(person_box),
                'object_box': list(object_box)
            })

        except (IndexError, KeyError, Exception):
            continue

    if not pair_actions:
        return []

    # Select one action per unique pair
    hoi_triplets = []
    for pair_key, actions in pair_actions.items():
        primary_action = select_primary_action(actions)
        hoi_triplets.append(primary_action)

    # Build deduplicated boxes
    for triplet in hoi_triplets:
        person_box_key = tuple(triplet['person_box'])
        if person_box_key not in box_map:
            box_map[person_box_key] = len(boxes)
            boxes.append(triplet['person_box'])

        object_box_key = tuple(triplet['object_box'])
        if object_box_key not in box_map:
            box_map[object_box_key] = len(boxes)
            boxes.append(triplet['object_box'])

    # Create referring instruction for each triplet
    for triplet in hoi_triplets:
        person_idx = box_map[tuple(triplet['person_box'])]
        object_idx = box_map[tuple(triplet['object_box'])]

        # Response: "action object" (e.g., "riding bicycle", "talking person")
        response = f"{triplet['action']} {triplet['object_name']}"

        referring_instruction = {
            "file_name": ann['file_name'],
            "width": ann.get('width', 640),
            "height": ann.get('height', 480),
            "boxes": boxes,
            "conversation": [
                {
                    "from": "human",
                    "value": REFERRING_QUERY_TEMPLATE,
                    "box_inds": [person_idx, object_idx]
                },
                {
                    "from": "gpt",
                    "value": response,
                    "box_inds": None
                }
            ]
        }
        instructions.append(referring_instruction)

    return instructions


def convert_swig_to_action_referring(
    anno_file: str,
    output_file: str,
    max_images: Optional[int] = None
) -> None:
    """
    Convert SWIG-HOI to action referring instruction format.

    Args:
        anno_file: Path to SWIG annotation JSON
        output_file: Path to output instruction JSON
        max_images: Maximum number of images to process (None for all)
    """

    print(f"\n{'='*80}")
    print(f"SWIG-HOI to Action Referring Converter")
    print(f"Output format: 'action object' (e.g., 'riding bicycle', 'talking person')")
    print(f"{'='*80}\n")

    print(f"Loading SWIG annotations from: {anno_file}")
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} images")

    print(f"Loading SWIG categories...")
    SWIG_ACTIONS, SWIG_CATEGORIES = load_swig_categories()
    action_id2name = {x["id"]: x["name"] for x in SWIG_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in SWIG_CATEGORIES}

    print(f"Loaded {len(action_id2name)} actions and {len(object_id2name)} objects")
    print(f"Note: Category ID 0 = 'person' (for person-to-person interactions)")

    if max_images is not None:
        annotations = annotations[:max_images]
        print(f"Processing subset of {len(annotations)} images")

    all_instructions = []
    stats = {
        'referring_samples': 0,
        'skipped': 0,
        'degenerate_boxes': 0,
        'person_to_person': 0
    }

    for idx, ann in enumerate(annotations):
        if not ann.get('hoi_annotations'):
            stats['skipped'] += 1
            continue

        instructions = convert_single_image_swig_action_referring(
            ann, action_id2name, object_id2name
        )

        if instructions:
            stats['referring_samples'] += len(instructions)

            # Count person-to-person interactions
            for inst in instructions:
                response = inst['conversation'][1]['value']
                if response.endswith(' person'):
                    stats['person_to_person'] += 1

            all_instructions.extend(instructions)
        else:
            stats['skipped'] += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(annotations)} images, "
                  f"generated {len(all_instructions)} samples...")

    print(f"\nSaving {len(all_instructions)} instruction examples to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_instructions, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Conversion Complete!")
    print(f"{'='*80}")
    print(f"✓ Total samples: {len(all_instructions)}")
    print(f"  - Referring samples: {stats['referring_samples']}")
    print(f"  - Person-to-person samples: {stats['person_to_person']} "
          f"({100*stats['person_to_person']/max(stats['referring_samples'],1):.1f}%)")
    print(f"  - Images processed: {len(annotations) - stats['skipped']}")
    print(f"  - Skipped: {stats['skipped']}")
    print(f"  - Output file: {output_file}")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for SWIG converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert SWIG-HOI to action referring format (action+object)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--anno-file',
        type=str,
        required=True,
        help='Path to SWIG annotation file (swig_train_1000.json or swig_test_1000.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to output instruction JSON file'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (None for all)'
    )

    args = parser.parse_args()

    if not Path(args.anno_file).exists():
        print(f"ERROR: Annotation file not found: {args.anno_file}")
        return 1

    convert_swig_to_action_referring(
        anno_file=args.anno_file,
        output_file=args.output_file,
        max_images=args.max_images
    )

    return 0


if __name__ == '__main__':
    exit(main())
