"""
HICO-DET to Groma Grounding Instruction Converter (Multi-Pair per Action)

Converts HICO-DET annotations to Groma's grounding instruction format where:
- Each sample represents ONE action type in an image
- Input prompt specifies the action and object
- Output provides ALL person-object pairs performing that specific action

New Format:
- Input: "[grounding] Identify the following person and objects in the image: person {action} {object} and the {object}"
- Output (multi-line for multiple pairs):
  <p>person</p><roi><r0></roi><p>{object}</p><roi><r1></roi>
  <p>person</p><roi><r2></roi><p>{object}</p><roi><r3></roi>
  ...

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
    """Normalize HICO object names to natural language (keep original dataset names)."""
    return object_name.replace('_', ' ')


def group_hoi_by_action_object(hoi_annotations, box_annotations, action_id2name, object_id2name):
    """
    Group HOI annotations by (action_id, object_name).

    For each unique (action, object) combination, collect all person-object pairs.
    This allows us to return ALL pairs performing the same action with the same object type.

    Returns:
        Dict mapping (action_id, object_name) -> list of (person_box, object_box, subject_id, object_id)
    """
    action_object_groups = defaultdict(list)

    for hoi in hoi_annotations:
        try:
            subject_id = hoi['subject_id']
            object_id = hoi['object_id']
            action_id = hoi['category_id'] - 1  # HICO actions are 1-indexed

            person_box = tuple(box_annotations[subject_id]['bbox'])
            object_box = tuple(box_annotations[object_id]['bbox'])
            object_category_id = box_annotations[object_id]['category_id']

            # Validate action and object category
            if action_id not in action_id2name or object_category_id not in object_id2name:
                continue

            # Get normalized names
            action_name = normalize_action_to_ing_form(action_id2name[action_id])
            object_name = normalize_object_name(object_id2name[object_category_id])

            # Group key: (action_id, object_name)
            # This ensures all "holding tennis racket" pairs are grouped together
            group_key = (action_id, object_name)

            # Deduplicate within group: skip if exact same (person_box, object_box) already exists
            pair = (person_box, object_box)
            existing_pairs = [(p[0], p[1]) for p in action_object_groups[group_key]]

            if pair not in existing_pairs:
                action_object_groups[group_key].append({
                    'person_box': list(person_box),
                    'object_box': list(object_box),
                    'subject_id': subject_id,
                    'object_id': object_id,
                    'action_name': action_name,
                    'object_name': object_name
                })

        except (IndexError, KeyError, Exception) as e:
            # Skip invalid annotations
            continue

    return action_object_groups


def convert_single_image_hico_grounding_only(
    ann: Dict,
    action_id2name: Dict,
    object_id2name: Dict
) -> List[Dict]:
    """
    Convert single HICO image to grounding instruction format.
    Generates ONE sample per unique (action, object) combination.
    Each sample contains ALL person-object pairs performing that action with that object.

    Args:
        ann: HICO annotation dict for one image
        action_id2name: Mapping from action ID to action name
        object_id2name: Mapping from object category ID to object name

    Returns:
        List of grounding instruction dicts (one per action-object combination)
    """

    box_annotations = ann['annotations']
    hoi_annotations = ann['hoi_annotation']

    # Group HOI annotations by (action, object)
    action_object_groups = group_hoi_by_action_object(
        hoi_annotations, box_annotations, action_id2name, object_id2name
    )

    if not action_object_groups:
        return []

    # Generate ONE sample per (action, object) group
    grounding_samples = []

    for group_idx, ((action_id, object_name), pairs) in enumerate(action_object_groups.items()):
        if not pairs:
            continue

        # Use first pair to get action name (all pairs in group have same action/object)
        action = pairs[0]['action_name']

        # Build focused prompt for THIS action-object combination
        prompt = f"[grounding] Identify the following person and objects in the image: person {action} {object_name} and the {object_name}"

        # Build response with ALL pairs in this group (multi-line)
        response_lines = []
        all_boxes = []
        box_inds = []

        for pair_idx, pair in enumerate(pairs):
            person_box = pair['person_box']
            object_box = pair['object_box']

            # Add boxes to list
            box_start_idx = len(all_boxes)
            all_boxes.append(person_box)
            all_boxes.append(object_box)

            # Build response line for this pair
            response_lines.append(
                f"<p>person</p><roi><r{box_start_idx}></roi><p>{object_name}</p><roi><r{box_start_idx + 1}></roi>"
            )
            box_inds.extend([box_start_idx, box_start_idx + 1])

        # Join all response lines with newline
        response = '\n'.join(response_lines)

        # Create grounding sample
        grounding_sample = {
            "file_name": ann['file_name'],
            "action_object_id": f"{ann['img_id']}_{action_id}_{object_name.replace(' ', '_')}",
            "width": ann.get('width', 640),
            "height": ann.get('height', 480),
            "boxes": all_boxes,  # All boxes (person1, object1, person2, object2, ...)
            "conversation": [
                {
                    "from": "human",
                    "value": prompt,
                    "box_inds": None
                },
                {
                    "from": "gpt",
                    "value": response,
                    "box_inds": box_inds  # [0, 1, 2, 3, ...]
                }
            ],
            # Metadata for analysis
            "action": action,
            "object_category": object_name,
            "num_pairs": len(pairs),
            "original_image_id": ann['img_id']
        }

        grounding_samples.append(grounding_sample)

    return grounding_samples


def convert_hico_to_grounding(
    anno_file: str,
    output_file: str,
    split: str = 'train',
    max_images: Optional[int] = None
) -> None:
    """
    Convert HICO-DET to grounding instruction format.
    Generates ONE sample per unique (action, object) combination per image.
    Each sample contains ALL pairs performing that specific action with that object.

    Args:
        anno_file: Path to HICO annotation JSON (trainval_hico_ann.json or test_hico_ann.json)
        output_file: Path to output grounding instruction JSON
        split: 'train' or 'test'
        max_images: Maximum number of images to process (None for all)
    """

    print(f"\n{'='*80}")
    print(f"HICO-DET to Groma Grounding Converter (Multi-Pair per Action)")
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

    all_grounding_samples = []
    stats = {
        'total_images': 0,
        'images_with_hoi': 0,
        'total_action_object_groups': 0,
        'total_pairs': 0,
        'skipped': 0
    }

    # Action distribution tracking
    action_counts = defaultdict(int)
    object_counts = defaultdict(int)
    pairs_per_sample = []

    for idx, ann in enumerate(annotations):
        if not ann.get('hoi_annotation'):
            stats['skipped'] += 1
            continue

        stats['total_images'] += 1

        # Generate grounding samples for this image (one per action-object combination)
        grounding_samples = convert_single_image_hico_grounding_only(
            ann, action_id2name, object_id2name
        )

        if grounding_samples:
            stats['images_with_hoi'] += 1
            stats['total_action_object_groups'] += len(grounding_samples)
            all_grounding_samples.extend(grounding_samples)

            # Track statistics
            for sample in grounding_samples:
                action_counts[sample['action']] += 1
                object_counts[sample['object_category']] += 1
                stats['total_pairs'] += sample['num_pairs']
                pairs_per_sample.append(sample['num_pairs'])
        else:
            stats['skipped'] += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(annotations)} images, "
                  f"generated {len(all_grounding_samples)} grounding samples...")

    print(f"\nSaving {len(all_grounding_samples)} grounding samples to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_grounding_samples, f, indent=2)

    # Print comprehensive statistics
    print(f"\n{'='*80}")
    print(f"Conversion Complete!")
    print(f"{'='*80}")
    print(f"✓ Total grounding samples: {len(all_grounding_samples)}")
    print(f"  - Images processed: {stats['total_images']}")
    print(f"  - Images with HOI: {stats['images_with_hoi']}")
    print(f"  - Unique (action, object) groups: {stats['total_action_object_groups']}")
    print(f"  - Total person-object pairs: {stats['total_pairs']}")
    if stats['images_with_hoi'] > 0:
        print(f"  - Avg groups per image: {stats['total_action_object_groups'] / stats['images_with_hoi']:.2f}")
    if pairs_per_sample:
        avg_pairs = sum(pairs_per_sample) / len(pairs_per_sample)
        max_pairs = max(pairs_per_sample)
        print(f"  - Avg pairs per sample: {avg_pairs:.2f}")
        print(f"  - Max pairs in single sample: {max_pairs}")
    print(f"  - Skipped: {stats['skipped']}")
    print(f"\n✓ Action distribution (top 10):")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {action}: {count}")
    print(f"\n✓ Object distribution (top 10):")
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {obj}: {count}")
    print(f"\n✓ Output file: {output_file}")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for HICO grounding converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert HICO-DET to Groma grounding format (multi-pair per action)',
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
        help='Path to output grounding instruction JSON file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='Dataset split'
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

    convert_hico_to_grounding(
        anno_file=args.anno_file,
        output_file=args.output_file,
        split=args.split,
        max_images=args.max_images
    )

    return 0


if __name__ == '__main__':
    exit(main())
