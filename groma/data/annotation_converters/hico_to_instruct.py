"""
HICO-DET to Groma Grounding + Referring Instruction Converter

Converts HICO-DET annotations to Groma's instruction format for:
1. Grounding task: Detect entities and describe interactions
2. Referring task: Recognize action given entity bounding boxes

All actions normalized to -ing form for consistency.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# Import HICO categories
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_hico_categories():
    """Load HICO categories from hico_categories.py file."""
    hico_cat_file = Path(__file__).parent.parent.parent / 'eval' / 'hoi_eval' / 'hico_categories.py'

    with open(hico_cat_file, 'r') as f:
        content = f.read()

    namespace = {}
    exec(content, namespace)

    return namespace['HICO_ACTIONS'], namespace['HICO_OBJECTS']


# Query templates for grounding task (with [grounding] prefix)
GROUNDING_QUERY_TEMPLATES = [
    "[grounding] Describe what each person is doing with objects individually. Focus on actions only.",
    "[grounding] What actions are the people performing with objects?",
    "[grounding] Describe the human-object interactions in this image.",
    "[grounding] What are people doing with the objects around them?",
    "[grounding] Identify all human-object interactions in this scene.",
]


def normalize_action_to_ing_form(action_name: str) -> str:
    """
    Normalize action names to -ing form for consistent training.
    Works for both HICO (root form) and SWIG (already -ing form).
    """
    action_name = action_name.strip().lower().replace('_', ' ')

    # If already in -ing form, return as-is
    if action_name.endswith('ing'):
        return action_name

    # Comprehensive mapping: root verb → -ing form
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
    """Normalize HICO object names to natural language."""
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


def convert_single_image_hico_both_tasks(
    ann: Dict,
    action_id2name: Dict,
    object_id2name: Dict,
    grounding_template_idx: int = 0
) -> List[Dict]:
    """
    Convert single HICO image to BOTH grounding and referring formats.

    Returns:
        List of instruction dicts (1 grounding + N referring)
    """

    instructions = []
    boxes = []
    box_map = {}

    box_annotations = ann['annotations']
    hoi_annotations = ann['hoi_annotation']

    # Group HOIs by (person, object) pair
    pair_actions = defaultdict(list)

    for hoi in hoi_annotations:
        try:
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']
            action_id = hoi['category_id'] - 1

            person_box = tuple(box_annotations[subject_id]['bbox'])
            object_box = tuple(box_annotations[object_id]['bbox'])
            object_category_id = box_annotations[object_id]['category_id']

            if action_id not in action_id2name or object_category_id not in object_id2name:
                continue

            # Normalize to -ing form
            action_name = normalize_action_to_ing_form(action_id2name[action_id])
            object_name = normalize_object_name(object_id2name[object_category_id])

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

    # ========================================
    # TASK 1: GROUNDING (Detection + Description)
    # ========================================
    grounding_response_parts = []
    grounding_box_inds = []

    for i, triplet in enumerate(hoi_triplets):
        person_idx = box_map[tuple(triplet['person_box'])]
        object_idx = box_map[tuple(triplet['object_box'])]

        grounding_box_inds.extend([person_idx, object_idx])

        if i == 0:
            sentence = f"The <p> person </p> <roi> <ground_box> </roi> is <p> {triplet['action']} </p> the <p> {triplet['object_name']} </p> <roi> <ground_box> </roi>."
        else:
            sentence = f" Another <p> person </p> <roi> <ground_box> </roi> is <p> {triplet['action']} </p> a <p> {triplet['object_name']} </p> <roi> <ground_box> </roi>."

        grounding_response_parts.append(sentence)

    grounding_response = "".join(grounding_response_parts).strip()

    # Grounding instruction
    grounding_instruction = {
        "file_name": ann['file_name'],
        "width": ann.get('width', 640),
        "height": ann.get('height', 480),
        "boxes": boxes,
        "conversation": [
            {
                "from": "human",
                "value": GROUNDING_QUERY_TEMPLATES[grounding_template_idx],
                "box_inds": None
            },
            {
                "from": "gpt",
                "value": grounding_response,
                "box_inds": grounding_box_inds
            }
        ]
    }
    instructions.append(grounding_instruction)

    # ========================================
    # TASK 2: REFERRING (Action Recognition)
    # ========================================
    for triplet in hoi_triplets:
        person_idx = box_map[tuple(triplet['person_box'])]
        object_idx = box_map[tuple(triplet['object_box'])]

        # Input: subject and object boxes using refer_box format
        referring_query = f"Describe only the action that <roi> <refer_box> </roi> <refer_feat> is doing to <roi> <refer_box> </roi> <refer_feat>. One action phrase:"

        referring_instruction = {
            "file_name": ann['file_name'],
            "width": ann.get('width', 640),
            "height": ann.get('height', 480),
            "boxes": boxes,
            "conversation": [
                {
                    "from": "human",
                    "value": referring_query,
                    "box_inds": [person_idx, object_idx]
                },
                {
                    "from": "gpt",
                    "value": triplet['action'],
                    "box_inds": None
                }
            ]
        }
        instructions.append(referring_instruction)

    return instructions


def convert_hico_to_grounding_and_referring(
    anno_file: str,
    output_file: str,
    num_grounding_variations: int = 1,
    split: str = 'trainval',
    max_images: Optional[int] = None
) -> None:
    """
    Convert HICO-DET to BOTH grounding and referring instruction formats.

    Args:
        anno_file: Path to HICO annotation JSON
        output_file: Path to output instruction JSON
        num_grounding_variations: Number of grounding template variations per image
        split: 'trainval' or 'test'
        max_images: Maximum number of images to process (None for all)
    """

    print(f"\n{'='*80}")
    print(f"HICO-DET to Groma Grounding + Referring Converter")
    print(f"{'='*80}\n")

    print(f"Loading HICO annotations from: {anno_file}")
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} images")

    print(f"Loading HICO categories...")
    HICO_ACTIONS, HICO_OBJECTS = load_hico_categories()
    action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}

    print(f"Loaded {len(action_id2name)} actions and {len(object_id2name)} objects")

    if max_images is not None:
        annotations = annotations[:max_images]
        print(f"Processing subset of {len(annotations)} images")

    all_instructions = []
    stats = {
        'grounding_samples': 0,
        'referring_samples': 0,
        'skipped': 0
    }

    for idx, ann in enumerate(annotations):
        if not ann.get('hoi_annotation'):
            stats['skipped'] += 1
            continue

        # Generate variations for grounding task
        for var_idx in range(num_grounding_variations):
            instructions = convert_single_image_hico_both_tasks(
                ann, action_id2name, object_id2name,
                grounding_template_idx=var_idx % len(GROUNDING_QUERY_TEMPLATES)
            )

            if instructions:
                # First is always grounding, rest are referring
                stats['grounding_samples'] += 1
                stats['referring_samples'] += len(instructions) - 1
                all_instructions.extend(instructions)
            else:
                if var_idx == 0:
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
    print(f"  - Grounding samples: {stats['grounding_samples']}")
    print(f"  - Referring samples: {stats['referring_samples']}")
    print(f"  - Images processed: {len(annotations) - stats['skipped']}")
    print(f"  - Skipped: {stats['skipped']}")
    print(f"  - Output file: {output_file}")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for HICO converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert HICO-DET to Groma grounding + referring format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--anno-file',
        type=str,
        required=True,
        help='Path to HICO annotation file (trainval_hico_ann.json)'
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
        default=1,
        help='Number of grounding template variations per image'
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

    convert_hico_to_grounding_and_referring(
        anno_file=args.anno_file,
        output_file=args.output_file,
        num_grounding_variations=args.num_variations,
        max_images=args.max_images
    )

    return 0


if __name__ == '__main__':
    exit(main())
