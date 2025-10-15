"""
HOI Grounding Dataset for Evaluation

Clean implementation without mmdet dependencies.
Manually handles image preprocessing to match what mmdet pipeline does.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


class HOIGroundingDataset(Dataset):
    """
    HOI Grounding evaluation dataset.

    Loads instruction-format data and preprocesses images to match
    mmdet's preprocessing (448x448 resize, normalize, to tensor).
    """

    def __init__(
        self,
        instruction_file,
        img_prefix,
        tokenizer,
        test_mode=True,
        conv_temp='llava'
    ):
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]

        # Load instruction data
        print(f"Loading HOI grounding instructions from: {instruction_file}")
        with open(instruction_file, 'r') as f:
            self.instructions = json.load(f)
        print(f"Loaded {len(self.instructions)} instruction examples")

        # Count total triplets
        total_triplets = 0
        for inst in self.instructions:
            box_inds = inst['conversation'][1]['box_inds']
            if box_inds:
                num_triplets = len(box_inds) // 2
                total_triplets += num_triplets
        print(f"Total HOI triplets: {total_triplets}")
        print(f"Average triplets per image: {total_triplets / len(self.instructions):.2f}")

        # Image preprocessing params (same as mmdet pipeline)
        self.img_scale = (448, 448)
        self.mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
        self.std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)

    def preprocess_image(self, img_path):
        """
        Preprocess image to match mmdet pipeline:
        1. Load image
        2. Resize to 448x448 (no ratio keeping)
        3. Convert RGB
        4. Normalize with mean/std
        5. Convert to tensor (C, H, W)
        """
        # Load image
        img = Image.open(img_path).convert('RGB')

        # Resize to 448x448
        img = img.resize(self.img_scale, Image.BILINEAR)

        # Convert to numpy array (H, W, C)
        img = np.array(img, dtype=np.float32)

        # Normalize
        img = (img - self.mean) / self.std

        # Convert to tensor and transpose to (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return img

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        """Get preprocessed image and instruction data"""
        inst = self.instructions[idx]

        # Load and preprocess image
        img_path = os.path.join(self.img_prefix, inst['file_name'])
        image = self.preprocess_image(img_path)  # (C, H, W) tensor

        # Get instruction data
        query = inst['conversation'][0]['value']
        gt_response = inst['conversation'][1]['value']
        gt_box_inds = inst['conversation'][1]['box_inds']
        gt_boxes = inst['boxes']  # [x, y, w, h] absolute pixels

        return {
            'image': image,
            'query': query,
            'gt_boxes': gt_boxes,
            'gt_box_inds': gt_box_inds,
            'gt_response': gt_response,
            'file_name': inst['file_name'],
            'width': inst['width'],
            'height': inst['height'],
            'idx': idx
        }


def extract_hoi_triplets(gt_boxes, gt_box_inds):
    """
    Extract HOI triplets from boxes and box indices.
    Each triplet is a pair of consecutive boxes (person, object).
    """
    if not gt_box_inds or len(gt_box_inds) == 0:
        return []

    triplets = []
    for i in range(0, len(gt_box_inds), 2):
        if i + 1 < len(gt_box_inds):
            person_idx = gt_box_inds[i]
            object_idx = gt_box_inds[i + 1]

            if person_idx < len(gt_boxes) and object_idx < len(gt_boxes):
                person_box = gt_boxes[person_idx]
                object_box = gt_boxes[object_idx]
                triplets.append((person_box, object_box))

    return triplets


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format"""
    # Convert to xyxy format
    def xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    # Calculate intersection
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union
    box1_area = box1[2] * box1[3]  # w * h
    box2_area = box2[2] * box2[3]  # w * h
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0.0

    return intersection / union
