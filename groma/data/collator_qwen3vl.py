"""
Data Collator for Original Qwen3-VL Training (no Groma components)

Simplified collator that only handles:
- input_ids, labels (text tokens)
- pixel_values, image_grid_thw (image features)

No region_images (DINOv2), no refer_boxes, no ground_boxes needed
since we're training the original Qwen3-VL without ROI components.
"""

import torch
import transformers
from dataclasses import dataclass

from groma.constants import IGNORE_INDEX


@dataclass
class DataCollatorForQwen3VL:
    """Data collator for original Qwen3-VL training.
    
    Handles batching of:
    - input_ids: Tokenized text sequences
    - labels: Training labels (with IGNORE_INDEX for masked positions)
    - pixel_values: Image features for Qwen3VL
    - image_grid_thw: Image grid dimensions
    - attention_mask: Attention mask for padding
    
    Unlike the Groma collator, this does NOT handle:
    - region_images (DINOv2 features)
    - refer_boxes (ROI coordinates for referring)
    - ground_boxes (output boxes for grounding)
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # Collect text inputs
        input_ids = [instance.get('input_ids', None) for instance in instances]
        labels = [instance.get('labels', None) for instance in instances]
        
        # Filter out None values
        valid_indices = [i for i, (ids, lbl) in enumerate(zip(input_ids, labels)) 
                        if ids is not None and lbl is not None and len(ids) > 0]
        
        if len(valid_indices) == 0:
            # Return empty batch
            return {
                'input_ids': torch.empty(0, 1, dtype=torch.long),
                'labels': torch.empty(0, 1, dtype=torch.long),
                'attention_mask': torch.empty(0, 1, dtype=torch.bool),
            }
        
        input_ids = [input_ids[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        instances = [instances[i] for i in valid_indices]
        
        # Image inputs for Qwen3VL
        pixel_values = [instance.get('pixel_values', None) for instance in instances]
        image_grid_thw = [instance.get('image_grid_thw', None) for instance in instances]
        
        # Handle image inputs - for Qwen3VL, these need to be processed
        # The dataset returns 'image' tensor from img_processor
        images = [instance.get('image', None) for instance in instances]
        
        # If pixel_values not provided directly, use images
        if all(pv is None for pv in pixel_values) and any(img is not None for img in images):
            # Stack images as pixel_values
            valid_images = [img for img in images if img is not None]
            if valid_images:
                pixel_values = torch.stack(valid_images)
            else:
                pixel_values = None
        elif all(pv is not None for pv in pixel_values):
            # Concatenate all pixel_values
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
            
        if image_grid_thw and all(x is not None for x in image_grid_thw):
            image_grid_thw = torch.stack(image_grid_thw)
        else:
            image_grid_thw = None

        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        # Truncate sequences to model's max position embeddings (2048 for Qwen3-VL)
        # Keep the LAST tokens to preserve assistant responses (most important for training)
        max_seq_length = 2048
        if input_ids.size(1) > max_seq_length:
            input_ids = input_ids[:, -max_seq_length:]
            labels = labels[:, -max_seq_length:]
            
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        
        # Only add image inputs if they exist
        if pixel_values is not None:
            batch['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            batch['image_grid_thw'] = image_grid_thw
            
        return batch

