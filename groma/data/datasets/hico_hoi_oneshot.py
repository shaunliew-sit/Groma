"""
HICO-DET One-Shot HOI Grounding Dataset for Evaluation

Task: Single-shot grounding of all HOI entities (persons + objects) in one model call
Input: "[grounding] Describe what each person is doing with objects individually. Focus on actions only."
Output: "The <p>person</p><roi><r9></roi> is <p>sitting on</p> the <p>bench</p><roi><r5></roi>."

This differs from hico_ground.py which makes separate queries per category.
"""

import os
import json
import torch
from collections import defaultdict

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


class HICOHOIOneShotTest:
    """
    HICO-DET One-Shot HOI Grounding evaluation dataset.

    Loads HICO test_hico_ann.json and extracts ALL boxes involved in HOI annotations
    for one-shot grounding evaluation (all entities detected in single model call).
    """

    # HICO uses COCO object categories (80 classes)
    # Mapping from COCO category_id to name
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
        73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
        78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
        84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }

    def __init__(
        self,
        ann_file,
        img_prefix,
        tokenizer,
        test_mode=True,
        conv_temp='llava'
    ):
        """
        Args:
            ann_file: Path to test_hico_ann.json
            img_prefix: Path to HICO images directory
            tokenizer: Tokenizer for text processing
            test_mode: Always True for evaluation
            conv_temp: Conversation template name
        """
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]

        # Load HICO annotations
        print(f"Loading HICO annotations from: {ann_file}")
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} images")

        # Build reverse mapping: name -> category_id
        self.name_to_cat_id = {v: k for k, v in self.COCO_CLASSES.items()}

        # Process each image to extract ALL boxes involved in HOI
        self._process_annotations()

        total_boxes = sum(len(item['all_boxes']) for item in self.processed_data)
        avg_boxes = total_boxes / len(self.processed_data)
        print(f"Total boxes across dataset: {total_boxes}")
        print(f"Average boxes per image: {avg_boxes:.2f}")

    def _process_annotations(self):
        """
        Process HICO annotations to extract ALL boxes involved in HOI annotations.

        For each image:
        1. Extract all boxes referenced in hoi_annotation (subject_id + object_id)
        2. Deduplicate boxes (same box may appear in multiple HOI pairs)
        3. Store box coordinates and category names
        """
        self.processed_data = []

        for item in self.data:
            # Build mapping from annotation index to box + category
            annotations = item['annotations']

            # Collect all unique annotation indices referenced in HOI pairs
            hoi_box_indices = set()
            for hoi in item.get('hoi_annotation', []):
                hoi_box_indices.add(hoi['subject_id'])
                hoi_box_indices.add(hoi['object_id'])

            # Extract boxes and categories for HOI-involved annotations
            all_boxes = []
            all_categories = []
            all_cat_ids = []

            for ann_idx in sorted(hoi_box_indices):
                if ann_idx >= len(annotations):
                    continue  # Skip invalid indices

                ann = annotations[ann_idx]
                cat_id = ann['category_id']
                bbox = ann['bbox']  # [x1, y1, x2, y2] format

                # Convert to [x, y, w, h] format (COCO format)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                bbox_xywh = [x1, y1, w, h]

                # Get category name
                if cat_id in self.COCO_CLASSES:
                    cat_name = self.COCO_CLASSES[cat_id]
                    all_boxes.append(bbox_xywh)
                    all_categories.append(cat_name)
                    all_cat_ids.append(cat_id)

            if len(all_boxes) == 0:
                continue  # Skip images with no valid HOI boxes

            processed_item = {
                'file_name': item['file_name'],
                'width': item['width'],
                'height': item['height'],
                'all_boxes': all_boxes,  # List of [x, y, w, h] boxes
                'all_categories': all_categories,  # List of category names
                'all_cat_ids': all_cat_ids,  # List of COCO category IDs
                'img_id': item['img_id'],
                'num_hoi_pairs': len(item.get('hoi_annotation', []))
            }

            self.processed_data.append(processed_item)

        print(f"Processed {len(self.processed_data)} images with HOI annotations")

    def build_prompt(self):
        """
        Build the one-shot HOI grounding prompt.
        Uses the exact prompt from training data.
        """
        # This is the exact prompt used in training data
        query = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

        # Build conversation following Groma's multi-turn format
        conversations = []

        # Turn 1: Initial context
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        # Turn 2: HOI grounding query
        conversations.append((self.conv_temp.roles[0], query))
        conversations.append((self.conv_temp.roles[1], ''))

        # Generate full prompt
        prompt = self.conv_temp.get_prompt(conversations)

        # Tokenize
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids)

        return input_ids, prompt

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get item for evaluation.

        Returns:
            Tuple of (all_boxes, all_categories, img_id, img_shape, file_name, base_item)
        """
        item = self.processed_data[idx]

        all_boxes = item['all_boxes']
        all_categories = item['all_categories']
        img_id = item['img_id']
        img_shape = (item['height'], item['width'])
        file_name = item['file_name']

        return all_boxes, all_categories, img_id, img_shape, file_name, item


def collate_fn(batch):
    """
    Custom collate function for dataloader.

    Since we process one image at a time, just return the single item.
    """
    # Batch size should be 1 for evaluation
    assert len(batch) == 1, "Batch size must be 1 for HOI one-shot evaluation"

    all_boxes, all_categories, img_id, img_shape, file_name, base_item = batch[0]

    return all_boxes, all_categories, img_id, img_shape, file_name, base_item
