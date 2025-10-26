"""
SWIG-HOI One-Shot HOI Grounding Dataset for Evaluation

Task: Single-shot grounding of all HOI entities (persons + objects) in one model call
Input: "[grounding] Describe what each person is doing with objects individually. Focus on actions only."
Output: "The <p>person</p><roi><r9></roi> is <p>sitting on</p> the <p>bench</p><roi><r5></roi>."

This differs from swig_ground.py which makes separate queries per category.
"""

import os
import json
import torch
from collections import defaultdict

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


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


class SWIGHOIOneShotTest:
    """
    SWIG-HOI One-Shot HOI Grounding evaluation dataset.

    Loads SWIG swig_test_1000.json and extracts ALL boxes involved in HOI annotations
    for one-shot grounding evaluation (all entities detected in single model call).
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
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]

        # Load SWIG category mapping (1000 object categories)
        self.cat_id_to_name = load_swig_categories()
        self.name_to_cat_id = {v: k for k, v in self.cat_id_to_name.items()}

        # Load SWIG annotations
        print(f"Loading SWIG annotations from: {ann_file}")
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} images")

        # Process each image to extract ALL boxes involved in HOI
        self._process_annotations()

        total_boxes = sum(len(item['all_boxes']) for item in self.processed_data)
        avg_boxes = total_boxes / len(self.processed_data) if self.processed_data else 0
        print(f"Total boxes across dataset: {total_boxes}")
        print(f"Average boxes per image: {avg_boxes:.2f}")

    def _process_annotations(self):
        """
        Process SWIG annotations to extract ALL boxes involved in HOI annotations.

        SWIG format:
        {
            "file_name": "tattooing_86.jpg",
            "hoi_annotations": [
                {"subject_id": 2, "object_id": 0, "action_id": 356},
                ...
            ],
            "box_annotations": [
                {"bbox": [x1, y1, x2, y2], "category_id": 83, "aux_category_id": [30]},
                ...
            ]
        }

        For each image:
        1. Extract all boxes referenced in hoi_annotations (subject_id + object_id)
        2. Deduplicate boxes (same box may appear in multiple HOI pairs)
        3. Store box coordinates and category names
        """
        self.processed_data = []

        for item in self.data:
            # Build mapping from box index to box annotation
            box_annotations = item['box_annotations']

            # Collect all unique box indices referenced in HOI pairs
            hoi_box_indices = set()
            for hoi in item.get('hoi_annotations', []):
                hoi_box_indices.add(hoi['subject_id'])
                hoi_box_indices.add(hoi['object_id'])

            # Extract boxes and categories for HOI-involved annotations
            all_boxes = []
            all_categories = []
            all_cat_ids = []

            for box_idx in sorted(hoi_box_indices):
                if box_idx >= len(box_annotations):
                    continue  # Skip invalid indices

                box_ann = box_annotations[box_idx]
                cat_id = box_ann['category_id']
                bbox = box_ann['bbox']  # [x1, y1, x2, y2] format

                # Convert to [x, y, w, h] format (COCO format)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                bbox_xywh = [x1, y1, w, h]

                # Get category name
                if cat_id in self.cat_id_to_name:
                    cat_name = self.cat_id_to_name[cat_id]
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
                'all_cat_ids': all_cat_ids,  # List of SWIG category IDs
                'img_id': item['img_id'],
                'num_hoi_pairs': len(item.get('hoi_annotations', []))
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
