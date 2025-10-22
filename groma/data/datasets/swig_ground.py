"""
SWIG-HOI Object Grounding Dataset for Evaluation

Uses swig_test_1000.json which has category_id mappings for all objects.
Loads SWIG categories from swig_v1_categories.py

Task: Referring Expression Comprehension (REC) for SWIG objects
Input: "Locate all {person} and {camera} in this image"
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


def load_swig_categories():
    """Load SWIG category mapping from swig_v1_categories.py"""
    import re

    # Read the category file
    categories_file = os.path.join(
        os.path.dirname(__file__),
        '../../eval/hoi_eval/swig_v1_categories.py'
    )

    with open(categories_file) as f:
        content = f.read()

    # Extract SWIG_CATEGORIES section (before SWIG_ACTIONS)
    categories_match = re.search(
        r'SWIG_CATEGORIES\s*=\s*\[(.*?)\]\s*SWIG_ACTIONS',
        content,
        re.DOTALL
    )

    if not categories_match:
        raise ValueError("Could not find SWIG_CATEGORIES in swig_v1_categories.py")

    categories_text = categories_match.group(1)

    # Extract id and name pairs
    category_pattern = r'"id":\s*(\d+),\s*"name":\s*"([^"]+)"'
    matches = re.findall(category_pattern, categories_text)

    # Create mapping
    cat_id_to_name = {int(cid): name for cid, name in matches}

    print(f"Loaded {len(cat_id_to_name)} SWIG categories")

    return cat_id_to_name


class SWIGGroundTest:
    """
    SWIG-HOI Object Grounding evaluation dataset.

    Loads SWIG swig_test_1000.json and extracts unique objects per image
    for grounding evaluation (similar to LVIS-Ground benchmark).
    """

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
            ann_file: Path to swig_test_1000.json
            img_prefix: Path to SWIG images directory
            tokenizer: Tokenizer for text processing
            test_mode: Always True for evaluation
            conv_temp: Conversation template name
        """
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.test_mode = test_mode
        self.conv_temp = conv_templates[conv_temp]

        # Load SWIG category mapping (1000 object categories)
        self.cat_id_to_name = load_swig_categories()
        self.name_to_cat_id = {v: k for k, v in self.cat_id_to_name.items()}

        # Load and process annotations
        self._load_annotations()

    def _load_annotations(self):
        """Load SWIG test annotations"""
        print(f"Loading SWIG annotations from: {self.ann_file}")

        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} images")

        # Process annotations
        self._process_annotations()

    def _process_annotations(self):
        """
        Process SWIG annotations to extract unique objects per image.

        SWIG format:
        {
            "file_name": "tattooing_86.jpg",
            "img_id": 2,
            "width": 771,
            "height": 512,
            "box_annotations": [
                {
                    "bbox": [x1, y1, x2, y2],  # xyxy format
                    "category_id": 83,
                    "aux_category_id": [30]
                },
                ...
            ]
        }
        """
        self.processed_data = []

        for item in self.data:
            # Group boxes by category_id
            category_boxes = defaultdict(list)

            for box_ann in item['box_annotations']:
                cat_id = box_ann['category_id']
                bbox = box_ann['bbox']  # [x1, y1, x2, y2]

                # Convert bbox from xyxy to xywh format for COCO evaluation
                x1, y1, x2, y2 = bbox
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

                # Map category_id to name
                if cat_id in self.cat_id_to_name:
                    cat_name = self.cat_id_to_name[cat_id]
                    category_boxes[cat_name].append(bbox_xywh)

            # Only include images with at least one recognized category
            if len(category_boxes) > 0:
                self.processed_data.append({
                    'img_id': item['img_id'],
                    'file_name': item['file_name'],
                    'width': item['width'],
                    'height': item['height'],
                    'categories': list(category_boxes.keys()),
                    'category_boxes': dict(category_boxes)
                })

        print(f"Processed {len(self.processed_data)} images with valid objects")

        # Calculate statistics
        all_categories = set()
        category_counts = []
        for item in self.processed_data:
            all_categories.update(item['categories'])
            category_counts.append(len(item['categories']))

        avg_categories = sum(category_counts) / len(category_counts) if category_counts else 0

        print(f"Total unique object categories: {len(all_categories)}")
        print(f"Categories: {sorted(list(all_categories))[:20]}...")
        print(f"Average unique categories per image: {avg_categories:.2f}")

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
    Custom collate function for SWIG Ground dataset.
    Enforces batch_size=1 for evaluation.
    """
    assert len(batch) == 1, "SWIG Ground evaluation requires batch_size=1"

    item = batch[0]
    return (
        item['categories'],
        item['category_boxes'],
        item['img_id'],
        item['img_shape'],
        item['file_name'],
        item['base_item']
    )
