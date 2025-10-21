"""
HICO-DET Object Grounding Dataset for Evaluation

Uses test_hico_ann.json which has clean category_id mappings for all objects.
No need to parse text or filter actions - just use the category IDs directly!

Task: Referring Expression Comprehension (REC) for HICO objects
Input: "Locate all {person} and {chair} in this image"
Output: Bounding boxes for each requested category
"""

import os
import json
import random
import torch
from collections import defaultdict

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


# Object grounding instruction templates (adapted from LVIS)
GROUNDING_INSTRUCTIONS = [
    "Locate all {} in this image.",
    "Identify all instances of {} in the photo.",
    "Find all instances of {} in the image.",
    "Point out all the {} visible in this picture.",
    "Detect and list each {} that appears in this photo.",
    "What is the position of each {} in the image?",
    "Where are the {} in this image?",
    "Show me all {} in this picture."
]


class HICOGroundTest:
    """
    HICO-DET Object Grounding evaluation dataset.

    Loads HICO test_hico_ann.json and extracts unique objects per image
    for grounding evaluation (similar to LVIS-Ground benchmark).
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

        # Process each image to extract unique objects and their boxes
        self._process_annotations()

        all_cats = self._get_all_categories()
        print(f"Total unique object categories: {len(all_cats)}")
        print(f"Categories: {sorted(all_cats)[:20]}{'...' if len(all_cats) > 20 else ''}")
        avg_objects = sum(len(item['categories']) for item in self.processed_data) / len(self.processed_data)
        print(f"Average unique categories per image: {avg_objects:.2f}")

    def _process_annotations(self):
        """
        Process HICO annotations to extract:
        1. Unique object categories per image (using category_id)
        2. Mapping from category name to ground truth boxes
        """
        self.processed_data = []

        for item in self.data:
            # Group boxes by category_id
            category_boxes = defaultdict(list)

            for ann in item['annotations']:
                cat_id = ann['category_id']
                bbox = ann['bbox']  # [x1, y1, x2, y2] format

                # Convert to [x, y, w, h] format
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                bbox_xywh = [x1, y1, w, h]

                # Get category name
                if cat_id in self.COCO_CLASSES:
                    cat_name = self.COCO_CLASSES[cat_id]
                    category_boxes[cat_name].append(bbox_xywh)

            if len(category_boxes) == 0:
                continue  # Skip images with no valid objects

            processed_item = {
                'file_name': item['file_name'],
                'width': item['width'],
                'height': item['height'],
                'categories': list(category_boxes.keys()),
                'category_boxes': dict(category_boxes),
                'img_id': item['img_id']
            }

            self.processed_data.append(processed_item)

        print(f"Processed {len(self.processed_data)} images with valid objects")

    def _get_all_categories(self):
        """Get set of all unique categories in dataset"""
        all_cats = set()
        for item in self.processed_data:
            all_cats.update(item['categories'])
        return all_cats

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get a single grounding query item.

        NOTE: This returns the base item data. The evaluation script
        will handle creating separate queries for each category.

        Returns dict with:
            - categories: List of object categories to ground
            - category_boxes: Ground truth boxes per category
            - img_id: Image ID
            - img_shape: (height, width)
            - file_name: Image filename
            - base_item: Full processed item for building prompts
        """
        item = self.processed_data[idx]

        return {
            'categories': item['categories'],
            'category_boxes': item['category_boxes'],
            'img_id': item['img_id'],
            'img_shape': (item['height'], item['width']),
            'file_name': item['file_name'],
            'base_item': item
        }

    def build_single_category_prompt(self, item, category):
        """
        Build a prompt for a single category query.

        Args:
            item: Processed data item
            category: Single category name to query

        Returns:
            Tokenized prompt for this category
        """
        # Create conversation following LVIS pattern
        conversations = []

        # Turn 1: Introduce image
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        # Turn 2: Query for THIS category only
        refexp = DEFAULT_TOKENS['boe'] + category + DEFAULT_TOKENS['eoe']
        instruct = random.choice(GROUNDING_INSTRUCTIONS).format(refexp)
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], ''))  # Empty response for model to fill

        prompt = self.conv_temp.get_prompt(conversations)

        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids

        return input_ids, prompt


def collate_fn(batch):
    """
    Custom collate function for HICO Ground dataset.
    Enforces batch_size=1 for evaluation.
    """
    assert len(batch) == 1, "HICO Ground evaluation requires batch_size=1"

    item = batch[0]
    return (
        item['categories'],
        item['category_boxes'],
        item['img_id'],
        item['img_shape'],
        item['file_name'],
        item['base_item']
    )
