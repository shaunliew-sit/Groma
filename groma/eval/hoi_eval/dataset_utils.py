"""Dataset utilities for loading and managing HOI datasets (HICO-DET, SWIG-HOI)."""

import os
import json
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS
from .swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES


def load_dataset(dataset_type, data_root, max_images=None):
    """Load dataset annotations"""
    if dataset_type == 'hico':
        return load_hico_dataset(data_root, max_images)
    elif dataset_type == 'swig':
        return load_swig_dataset(data_root, max_images)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_hico_dataset(data_root, max_images=None):
    """Load HICO-DET test dataset

    Args:
        data_root: Root directory of HICO dataset
        max_images: Maximum number of images to load (None for full dataset)
    """
    test_ann_file = os.path.join(data_root, 'annotations', 'test_hico_ann.json')
    test_img_dir = os.path.join(data_root, 'images', 'test2015')

    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"HICO test annotations not found: {test_ann_file}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"HICO test images not found: {test_img_dir}")

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    print(f"DEBUG: Found {len(annotations)} total annotations in HICO test file")

    # Limit to max_images if specified
    if max_images is not None:
        annotations = annotations[:max_images]
        print(f"DEBUG: Using subset of {len(annotations)} images for evaluation")

    dataset = []
    for ann in annotations:
        img_path = os.path.join(test_img_dir, ann['file_name'])
        if os.path.exists(img_path):
            # Include ALL annotation data in the dataset item
            dataset.append({
                'image_id': ann['img_id'],
                'image_path': img_path,
                'file_name': ann['file_name'],
                'width': ann.get('width', 640),
                'height': ann.get('height', 480),
                # CRITICAL: Include ground truth annotations
                'annotations': ann.get('annotations', []),
                'hoi_annotation': ann.get('hoi_annotation', [])
            })

    print(f"DEBUG: Successfully loaded {len(dataset)} valid HICO images")
    return dataset


def load_swig_dataset(data_root, max_images=None):
    """Load SWIG-HOI test dataset

    Args:
        data_root: Root directory of SWIG dataset
        max_images: Maximum number of images to load (None for full dataset)
    """
    test_ann_file = os.path.join(data_root, 'annotations', 'swig_test_1000.json')
    test_img_dir = os.path.join(data_root, 'images_512')

    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"SWIG test annotations not found: {test_ann_file}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"SWIG test images not found: {test_img_dir}")

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    print(f"DEBUG: Found {len(annotations)} total annotations in SWIG test file")

    # Limit to max_images if specified
    if max_images is not None:
        annotations = annotations[:max_images]
        print(f"DEBUG: Using subset of {len(annotations)} images for evaluation")

    dataset = []
    for ann in annotations:
        img_path = os.path.join(test_img_dir, ann['file_name'])
        if os.path.exists(img_path):
            # Include ALL annotation data in the dataset item
            dataset.append({
                'image_id': ann['img_id'],
                'image_path': img_path,
                'file_name': ann['file_name'],
                'width': ann.get('width', 512),
                'height': ann.get('height', 512),
                # CRITICAL: Include ground truth annotations
                'box_annotations': ann.get('box_annotations', []),
                'hoi_annotations': ann.get('hoi_annotations', [])
            })

    print(f"DEBUG: Successfully loaded {len(dataset)} valid SWIG images")
    return dataset


def find_image_in_dataset(image_path, dataset_type, data_root):
    """Find image information and ground truth in dataset"""
    image_name = os.path.basename(image_path)

    if dataset_type == 'hico':
        return find_hico_image_gt(image_name, data_root)
    elif dataset_type == 'swig':
        return find_swig_image_gt(image_name, data_root)
    else:
        return None


def find_hico_image_gt(image_name, data_root):
    """Find HICO image ground truth"""
    test_ann_file = os.path.join(data_root, 'annotations', 'test_hico_ann.json')

    if not os.path.exists(test_ann_file):
        print(f"WARNING: HICO annotations not found: {test_ann_file}")
        return None

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    print(f"DEBUG: Loaded {len(annotations)} annotations from HICO test set")
    print(f"DEBUG: Looking for image: {image_name}")

    # Find the specific image
    for i, ann in enumerate(annotations):
        if ann['file_name'] == image_name:
            print(f"DEBUG: Found image at index {i}")
            print(f"DEBUG: Image data keys: {ann.keys()}")

            result = {
                'image_id': ann['img_id'],
                'file_name': ann['file_name'],
                'width': ann.get('width', 640),
                'height': ann.get('height', 480),
                'annotations': ann.get('annotations', []),
                'hoi_annotation': ann.get('hoi_annotation', [])
            }

            print(f"DEBUG: Found {len(result['annotations'])} box annotations")
            print(f"DEBUG: Found {len(result['hoi_annotation'])} HOI annotations")

            # Show sample box annotation structure
            if result['annotations']:
                print(f"DEBUG: Sample box annotation: {result['annotations'][0]}")

            # Show sample HOI annotation structure
            if result['hoi_annotation']:
                print(f"DEBUG: Sample HOI annotation: {result['hoi_annotation'][0]}")

            return result

    print(f"WARNING: Image {image_name} not found in HICO test set")
    print(f"DEBUG: Available images (first 5): {[ann['file_name'] for ann in annotations[:5]]}")
    return None


def find_swig_image_gt(image_name, data_root):
    """Find SWIG image ground truth"""
    test_ann_file = os.path.join(data_root, 'annotations', 'swig_test_1000.json')

    if not os.path.exists(test_ann_file):
        print(f"WARNING: SWIG annotations not found: {test_ann_file}")
        return None

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    # Find the specific image
    for ann in annotations:
        if ann['file_name'] == image_name:
            return {
                'image_id': ann['img_id'],
                'file_name': ann['file_name'],
                'width': ann.get('width', 512),
                'height': ann.get('height', 512),
                'box_annotations': ann.get('box_annotations', []),
                'hoi_annotations': ann.get('hoi_annotations', [])
            }

    print(f"WARNING: Image {image_name} not found in SWIG test set")
    return None


def extract_ground_truth_hois(gt_data, dataset_type):
    """Extract ground truth HOI triplets from annotation data"""
    if dataset_type == 'hico':
        return extract_hico_ground_truth(gt_data)
    elif dataset_type == 'swig':
        return extract_swig_ground_truth(gt_data)
    else:
        return []


def extract_hico_ground_truth(gt_data):
    """Extract HICO ground truth HOI triplets"""
    if not gt_data:
        return []

    action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
    # HICO_INTERACTIONS uses string names, create (action_name, object_name) -> hoi_id mapper
    hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    gt_hois = []
    box_annos = gt_data.get('annotations', [])
    hoi_annos = gt_data.get('hoi_annotation', [])

    print(f"DEBUG: Processing {len(hoi_annos)} HOI annotations")

    for i, hoi in enumerate(hoi_annos):
        try:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            # HICO action categories start from 1, so we subtract 1 for 0-based indexing
            action_id = hoi["category_id"] - 1
            # HICO object categories - let's check both with and without offset
            object_category_id_raw = box_annos[hoi["object_id"]]["category_id"]
            print(f"DEBUG: Raw object_category_id from box annotation: {object_category_id_raw}")

            # Try both raw and adjusted object ID to see which one exists
            object_id_adjusted = object_category_id_raw - 1 if object_category_id_raw > 0 else object_category_id_raw

            print(f"DEBUG: HOI {i+1}: action_id={action_id}, raw_object_id={object_category_id_raw}, adjusted_object_id={object_id_adjusted}")

            # Check if action_id is valid
            if action_id not in action_id2name:
                print(f"WARNING: Invalid action_id {action_id}, skipping HOI annotation {i+1}")
                continue

            # Check which object ID format works
            object_id = None
            if object_category_id_raw in object_id2name:
                object_id = object_category_id_raw
                print(f"DEBUG: Using raw object_id: {object_id}")
            elif object_id_adjusted in object_id2name:
                object_id = object_id_adjusted
                print(f"DEBUG: Using adjusted object_id: {object_id}")
            else:
                print(f"WARNING: Neither raw ({object_category_id_raw}) nor adjusted ({object_id_adjusted}) object_id found in object mapping, skipping HOI annotation {i+1}")
                continue

            action_name = action_id2name[action_id]
            object_name = object_id2name[object_id]

            print(f"DEBUG: action_name='{action_name}', object_name='{object_name}'")

            # Look up HOI ID using action and object names
            hoi_key = (action_name, object_name)
            if hoi_key not in hoi_mapper:
                print(f"WARNING: No HOI mapping found for ('{action_name}', '{object_name}'), skipping")
                print(f"DEBUG: Available action-object combinations for '{action_name}': {[k for k in hoi_mapper.keys() if k[0] == action_name]}")
                continue

            hoi_id = hoi_mapper[hoi_key]
            print(f"DEBUG: Found hoi_id={hoi_id} for ('{action_name}', '{object_name}')")

            gt_hois.append({
                'hoi_id': hoi_id,
                'action_name': action_name,
                'object_name': object_name,
                'person_bbox': person_box,
                'object_bbox': object_box,
                'subject_id': hoi["subject_id"],
                'object_id': hoi["object_id"]
            })

        except Exception as e:
            print(f"ERROR: Failed to process HOI annotation {i+1}: {str(e)}")
            print(f"  HOI data: {hoi}")
            continue

    print(f"DEBUG: Successfully extracted {len(gt_hois)} valid HOI annotations")
    return gt_hois


def extract_swig_ground_truth(gt_data):
    """Extract SWIG ground truth HOI triplets"""
    if not gt_data:
        return []

    hoi_mapper = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

    # Create ID to name mappings
    action_id2name = {x["id"]: x["name"] for x in SWIG_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in SWIG_CATEGORIES}

    gt_hois = []
    box_annos = gt_data.get('box_annotations', [])
    hoi_annos = gt_data.get('hoi_annotations', [])

    print(f"DEBUG: Processing {len(hoi_annos)} SWIG HOI annotations")

    for i, hoi in enumerate(hoi_annos):
        try:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            action_id = hoi["action_id"]
            object_id = box_annos[hoi["object_id"]]["category_id"]

            print(f"DEBUG: SWIG HOI {i+1}: action_id={action_id}, object_id={object_id}")

            # Get action and object names
            action_name = action_id2name.get(action_id, f"action_{action_id}")
            object_name = object_id2name.get(object_id, f"object_{object_id}")

            print(f"DEBUG: SWIG HOI {i+1}: action_name='{action_name}', object_name='{object_name}'")

            hoi_id = hoi_mapper.get((action_id, object_id))
            if hoi_id is not None:
                gt_hois.append({
                    'hoi_id': hoi_id,
                    'action_id': action_id,
                    'object_id': object_id,
                    'action_name': action_name,
                    'object_name': object_name,
                    'person_bbox': person_box,
                    'object_bbox': object_box,
                    'subject_id': hoi["subject_id"],
                    'object_id_idx': hoi["object_id"]
                })
                print(f"DEBUG: SWIG HOI {i+1}: hoi_id={hoi_id} for ('{action_name}', '{object_name}')")
            else:
                print(f"WARNING: No SWIG HOI mapping found for (action_id={action_id}, object_id={object_id})")

        except Exception as e:
            print(f"ERROR: Failed to process SWIG HOI annotation {i+1}: {str(e)}")
            print(f"  HOI data: {hoi}")
            continue

    print(f"DEBUG: Successfully extracted {len(gt_hois)} valid SWIG HOI annotations")
    return gt_hois