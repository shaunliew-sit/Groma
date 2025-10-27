"""
HICO-DET Action Referring Dataset for Evaluation

Task: Given two bounding boxes (person + object), predict the connecting action
Input: Person bbox + Object bbox + query "Describe only the action that <roi> is doing to <roi>"
Output: Action phrase (e.g., "sit on", "hold", "ride")

Format: COCO caption format where each triplet is treated as separate "image"
Metrics: METEOR, CIDEr (semantic similarity for action phrases)
"""

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor

from pycocotools.coco import COCO
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


# Action referring query templates
ACTION_REFERRING_TEMPLATES = [
    "Describe only the action that {} is doing to {}. One action phrase:",
    "What action is {} performing on {}? Answer with one phrase:",
    "How is {} interacting with {}? One action:",
    "What is {} doing to {}? Single action phrase:",
]


class HICOActionReferringTest:
    """
    HICO-DET Action Referring evaluation dataset.

    Each sample represents one HOI triplet (person, object, action).
    The dataset is pre-processed into COCO caption format where:
    - Each "image" is actually a unique triplet ID
    - Each "caption" is the ground truth action phrase
    - Metadata includes person/object bboxes and categories
    """

    def __init__(
        self,
        ann_file,
        img_prefix,
        tokenizer,
        vis_processor=None,
        test_mode=True,
        conv_temp='llava',
        query_template_idx=0
    ):
        """
        Args:
            ann_file: Path to COCO-format action referring annotations (from prepare script)
            img_prefix: Path to HICO images directory
            tokenizer: Tokenizer for text processing
            vis_processor: Image processor (AutoImageProcessor)
            test_mode: Always True for evaluation
            conv_temp: Conversation template name (default: 'llava')
            query_template_idx: Which template to use from ACTION_REFERRING_TEMPLATES (default: 0)
        """
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]
        self.query_template = ACTION_REFERRING_TEMPLATES[query_template_idx]

        # Initialize visual processor
        if vis_processor is None:
            # Auto-detect from tokenizer path if not provided
            model_path = os.path.dirname(tokenizer.name_or_path) if hasattr(tokenizer, 'name_or_path') else None
            if model_path and os.path.exists(model_path):
                vis_processor = AutoImageProcessor.from_pretrained(model_path)
            else:
                raise ValueError("vis_processor must be provided or tokenizer must have valid model path")
        self.vis_processor = vis_processor

        # Load COCO-format annotations
        print(f"Loading HICO Action Referring annotations from: {ann_file}")
        self.coco = COCO(ann_file)

        # Get all triplet IDs (each triplet is treated as separate "image")
        self.triplet_ids = self.coco.getImgIds()
        print(f"Loaded {len(self.triplet_ids)} HOI triplets for evaluation")

        # Build data_infos for compatibility
        self.data_infos = [self.coco.loadImgs([tid])[0] for tid in self.triplet_ids]

        # Get unique original images
        unique_imgs = set([info['original_image_id'] for info in self.data_infos])
        print(f"Unique original images: {len(unique_imgs)}")

        # Get action distribution
        self._print_action_distribution()

    def _print_action_distribution(self):
        """Print distribution of actions in dataset"""
        from collections import Counter

        ann_ids = self.coco.getAnnIds()
        anns = self.coco.loadAnns(ann_ids)
        action_counter = Counter([ann['caption'] for ann in anns])

        print(f"\nTop-10 most frequent actions:")
        for action, count in action_counter.most_common(10):
            print(f"  {action:<20s}: {count:4d} samples")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """
        Get one HOI triplet for evaluation.

        Returns:
            dict with:
                - input_ids: Tokenized query
                - image: Processed image tensor
                - triplet_id: Unique triplet identifier
                - person_bbox: Normalized person bbox [cx, cy, w, h]
                - object_bbox: Normalized object bbox [cx, cy, w, h]
                - gt_action: Ground truth action phrase
                - img_info: Full image metadata
        """
        img_info = self.data_infos[idx]
        triplet_id = img_info['id']

        # Load and process image
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Resize to 448x448 (required by Groma)
        image = image.resize((448, 448))
        image_tensor = self.vis_processor.preprocess(image, return_tensors='pt')['pixel_values'].squeeze(0)

        # Get bboxes (stored as absolute pixels in [x, y, w, h] format)
        person_bbox_abs = img_info['subject_bbox']  # [x, y, w, h] absolute
        object_bbox_abs = img_info['object_bbox']   # [x, y, w, h] absolute

        # Normalize bboxes to [cx, cy, w, h] in [0, 1] range
        width, height = img_info['width'], img_info['height']
        person_bbox_norm = self._normalize_bbox(person_bbox_abs, width, height)
        object_bbox_norm = self._normalize_bbox(object_bbox_abs, width, height)

        # Build conversation query
        query = self._build_query()
        input_ids = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids.squeeze(0)

        # Get ground truth action
        ann_ids = self.coco.getAnnIds(imgIds=[triplet_id])
        anns = self.coco.loadAnns(ann_ids)
        gt_action = anns[0]['caption'] if len(anns) > 0 else "unknown"

        return {
            'input_ids': input_ids,
            'image': image_tensor,
            'triplet_id': triplet_id,
            'person_bbox': person_bbox_norm,
            'object_bbox': object_bbox_norm,
            'gt_action': gt_action,
            'img_info': img_info
        }

    def _normalize_bbox(self, bbox, width, height):
        """
        Convert absolute bbox [x, y, w, h] to normalized [cx, cy, w, h].

        Args:
            bbox: [x, y, w, h] in absolute pixels
            width, height: Image dimensions

        Returns:
            torch.Tensor of shape (4,) with [cx, cy, w, h] normalized to [0, 1]
        """
        x, y, w, h = bbox

        # Convert to center coordinates and normalize
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height

        return torch.tensor([cx, cy, w_norm, h_norm], dtype=torch.float32)

    def _build_query(self):
        """
        Build action referring query with region tokens.

        Format:
            USER: Here is an image with region crops from it. Image: <image>. Regions: <region>.
            ASSISTANT: Thank you for the image! How can I assist you with it?
            USER: Describe only the action that <roi><refer_box></roi><refer_feat> is doing to <roi><refer_box></roi><refer_feat>. One action phrase:
            ASSISTANT: [model generates action here]

        Returns:
            Full prompt string ready for tokenization
        """
        conversations = []

        # Initial greeting
        instruct = f"Here is an image with region crops from it. "
        instruct += f"Image: {DEFAULT_TOKENS['image']}. "
        instruct += f"Regions: {DEFAULT_TOKENS['region']}."
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        # Action query with TWO region tokens (subject and object)
        # CRITICAL: Use placeholder tokens with SPACES (matching training format)
        # Training format: "<roi> <refer_box> </roi> <refer_feat>"
        subject_ref = (DEFAULT_TOKENS['bor'] + ' ' + DEFAULT_TOKENS['rbox'] + ' ' +
                      DEFAULT_TOKENS['eor'] + ' ' + DEFAULT_TOKENS['rfeat'])
        object_ref = (DEFAULT_TOKENS['bor'] + ' ' + DEFAULT_TOKENS['rbox'] + ' ' +
                     DEFAULT_TOKENS['eor'] + ' ' + DEFAULT_TOKENS['rfeat'])

        # Format query with template
        instruct = self.query_template.format(subject_ref, object_ref)

        # Add to conversation (empty answer for generation)
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], ''))

        # Get full prompt
        prompt = self.conv_temp.get_prompt(conversations)

        return prompt

    def get_img_info(self, triplet_id):
        """
        Get image info by triplet ID.

        Args:
            triplet_id: Unique triplet identifier

        Returns:
            Image info dict
        """
        return self.coco.loadImgs([triplet_id])[0]


def collate_fn(batch):
    """
    Custom collate function for action referring dataset.

    Since batch_size=1 for generation, just return the single item as-is
    but ensure tensors have batch dimension.

    Args:
        batch: List of dicts from __getitem__

    Returns:
        Dict with batched tensors
    """
    assert len(batch) == 1, "Action referring evaluation only supports batch_size=1"

    item = batch[0]

    return {
        'input_ids': item['input_ids'].unsqueeze(0),  # (1, seq_len)
        'image': item['image'].unsqueeze(0),          # (1, 3, 448, 448)
        'triplet_id': item['triplet_id'],             # str
        'person_bbox': item['person_bbox'],           # (4,)
        'object_bbox': item['object_bbox'],           # (4,)
        'gt_action': item['gt_action'],               # str
        'img_info': item['img_info']                  # dict
    }
