"""
SWIG-HOI Grounding Dataset (Multi-Pair per Action-Object)

Dataset for training and evaluation on SWIG-HOI grounding task.
Each sample represents ONE action-object combination with ALL matching pairs.
Supports person-person interactions.

Task: Given "[grounding] Identify the following person and objects in the image: person {action} {object} and the {object}."
Output: Multi-line format for multiple pairs:
<p>person</p><roi><r0></roi><p>{object}</p><roi><r1></roi>
<p>person</p><roi><r2></roi><p>{object}</p><roi><r3></roi>
...

For person-person: "[grounding] Identify the following person and objects in the image: person talking person and the person."
Output: <p>person</p><roi><r0></roi><p>person</p><roi><r1></roi>
"""

import os
import json
import torch
from PIL import Image
from typing import Dict, List, Optional

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


class SWIGGroundTrain:
    """
    SWIG-HOI Grounding training dataset.
    Each sample = one action-object combination with ALL matching person-object pairs.
    Multiple pairs in same sample are formatted as multi-line responses.
    Supports person-person interactions.
    """

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        tokenizer,
        vis_processor=None,
        test_mode: bool = False,
        conv_temp: str = 'llava'
    ):
        """
        Args:
            ann_file: Path to grounding instruction JSON (swig_ground_train.json)
            img_prefix: Path to SWIG images directory
            tokenizer: Tokenizer for text processing
            vis_processor: Image processor (optional)
            test_mode: If True, used for evaluation
            conv_temp: Conversation template name
        """
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.vis_processor = vis_processor
        self.test_mode = test_mode
        self.conv_temp = conv_templates[conv_temp]

        # Load grounding samples (one per HOI triplet)
        print(f"Loading SWIG grounding samples from: {ann_file}")
        with open(ann_file, 'r') as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} grounding samples")

        # Extract statistics
        unique_images = len(set(s['original_image_id'] for s in self.samples))
        print(f"Unique images: {unique_images}")
        if unique_images > 0:
            print(f"Avg triplets per image: {len(self.samples) / unique_images:.2f}")

        # Action and object distribution
        from collections import Counter
        actions = [s['action'] for s in self.samples]
        objects = [s['object_category'] for s in self.samples]
        action_counts = Counter(actions)
        object_counts = Counter(objects)

        # Person-person interaction stats
        person_person_count = sum(1 for s in self.samples if s.get('is_person_person', False))
        print(f"Person-person interactions: {person_person_count} / {len(self.samples)}")

        print(f"Unique actions: {len(action_counts)}")
        print(f"Unique object categories: {len(object_counts)}")
        print(f"Top 5 actions: {action_counts.most_common(5)}")
        print(f"Top 5 objects: {object_counts.most_common(5)}")

        # Image cache for efficiency (same image used for multiple triplets)
        self.image_cache = {}
        self.max_cache_size = 100  # Cache up to 100 images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one action-object group sample.

        Returns dict with:
            - input_ids: Tokenized input
            - image: Processed image tensor
            - boxes: List of boxes [person1_box, object1_box, person2_box, object2_box, ...]
            - box_inds: Indices for all boxes (e.g., [0,1,2,3,...] for multiple pairs)
            - metadata: action, object_category, is_person_person, etc.
        """
        sample = self.samples[idx]

        # Load image (with caching)
        img_path = os.path.join(self.img_prefix, sample['file_name'])

        if img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            image = Image.open(img_path).convert('RGB')

            # Add to cache if space available
            if len(self.image_cache) < self.max_cache_size:
                self.image_cache[img_path] = image
            else:
                # Simple cache replacement: remove first item
                first_key = next(iter(self.image_cache))
                del self.image_cache[first_key]
                self.image_cache[img_path] = image

        # Process image
        if self.vis_processor is not None:
            image_processed = self.vis_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            # Default processing
            import numpy as np
            image_resized = image.resize((448, 448), Image.BILINEAR)
            image_np = np.array(image_resized, dtype=np.float32)
            # Normalize
            mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
            std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
            image_np = (image_np - mean) / std
            # Convert to tensor (C, H, W)
            image_processed = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        # Build conversation from sample
        conversation = sample['conversation']

        # Tokenize conversation
        input_ids = self._tokenize_conversation(conversation)

        return {
            'input_ids': input_ids,
            'image': image_processed,
            'boxes': torch.tensor(sample['boxes'], dtype=torch.float32),
            'box_inds': sample['conversation'][1]['box_inds'],
            'file_name': sample['file_name'],
            'action_object_id': sample['action_object_id'],
            'action': sample['action'],
            'object_category': sample['object_category'],
            'is_person_person': sample.get('is_person_person', False),
            'original_image_id': sample['original_image_id']
        }

    def _tokenize_conversation(self, conversation):
        """Tokenize conversation with special tokens."""
        # Build conversation as list of (role, content) tuples
        # First add image/region initialization (required by model)
        conversations = []

        # Add image and region tokens prefix
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        # Add actual conversation
        for turn in conversation:
            role = turn['from']
            content = turn['value']

            if role == 'human':
                # Add grounding prefix token
                content = DEFAULT_TOKENS['ground'] + content
                conversations.append((self.conv_temp.roles[0], content))
            else:
                # Add separator tokens
                content = DEFAULT_TOKENS['sep'] + content + DEFAULT_TOKENS['sep']
                conversations.append((self.conv_temp.roles[1], content))

        # Get prompt
        prompt = self.conv_temp.get_prompt(conversations)

        # Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        ).input_ids[0]

        return input_ids


class SWIGGroundTest(SWIGGroundTrain):
    """
    SWIG-HOI Grounding test dataset.
    Inherits from SWIGGroundTrain but used for evaluation.
    """

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        tokenizer,
        vis_processor=None,
        conv_temp: str = 'llava'
    ):
        super().__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            tokenizer=tokenizer,
            vis_processor=vis_processor,
            test_mode=True,
            conv_temp=conv_temp
        )

    def __getitem__(self, idx):
        """
        Load one action-object group sample for evaluation.

        Returns same as training but with additional info for evaluation.
        Note: For evaluation, ground truth boxes are extracted and returned separately.
        """
        sample = self.samples[idx]

        # Load image (with caching)
        img_path = os.path.join(self.img_prefix, sample['file_name'])

        if img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            image = Image.open(img_path).convert('RGB')

            # Add to cache
            if len(self.image_cache) < self.max_cache_size:
                self.image_cache[img_path] = image
            else:
                # Simple cache replacement
                first_key = next(iter(self.image_cache))
                del self.image_cache[first_key]
                self.image_cache[img_path] = image

        # Process image
        if self.vis_processor is not None:
            image_processed = self.vis_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            # Default processing
            import numpy as np
            image_resized = image.resize((448, 448), Image.BILINEAR)
            image_np = np.array(image_resized, dtype=np.float32)
            # Normalize
            mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
            std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
            image_np = (image_np - mean) / std
            # Convert to tensor (C, H, W)
            image_processed = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        # Build conversation (only query for evaluation)
        query = sample['conversation'][0]['value']  # Human query

        # Tokenize query as list of (role, content) tuples
        # First add image/region initialization (required by model)
        conversations = []

        # Add image and region tokens prefix
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        # Add grounding query with prefix token
        query = DEFAULT_TOKENS['ground'] + query
        conversations.append((self.conv_temp.roles[0], query))
        conversations.append((self.conv_temp.roles[1], ''))

        prompt = self.conv_temp.get_prompt(conversations)

        input_ids = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        ).input_ids[0]

        # Ground truth boxes (all pairs)
        gt_boxes = sample['boxes']
        num_pairs = sample['num_pairs']

        return {
            'input_ids': input_ids,
            'image': image_processed,
            'file_name': sample['file_name'],
            'img_path': img_path,
            'action_object_id': sample['action_object_id'],
            'action': sample['action'],
            'object_category': sample['object_category'],
            'is_person_person': sample.get('is_person_person', False),
            'original_image_id': sample['original_image_id'],
            'width': sample['width'],
            'height': sample['height'],
            # Ground truth
            'gt_boxes': gt_boxes,  # All boxes for all pairs
            'num_pairs': num_pairs,
            'gt_response': sample['conversation'][1]['value']  # Expected multi-line response
        }


def collate_fn(batch):
    """
    Collate function for dataloader.

    For training: batches multiple triplets together
    For evaluation: typically batch_size=1
    """
    # For simplicity, return batch as-is (list of dicts)
    # Model will handle batching internally
    if len(batch) == 1:
        return batch[0]
    else:
        return batch
