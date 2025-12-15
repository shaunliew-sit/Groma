"""
[GROMA-QWEN V3] Dataset processor with Interaction Token Support

This module extends the V2 dataset processor to add interaction boxes
to the prompts for training the interaction-aware model.

Key Changes from V2:
--------------------
1. Adds interaction box (union of person + object) to prompts
2. Creates 3 refer_boxes instead of 2
3. Augments prompt text with interaction region description

Dataset Format (same as V2):
----------------------------
{
    "file_name": "image.jpg",
    "width": 640,
    "height": 480,
    "boxes": [[x1, y1, x2, y2], ...],
    "task_type": "referring" | "grounding",
    "conversation": [
        {"from": "human", "value": "...", "box_inds": [0, 1]},
        {"from": "gpt", "value": "...", "box_inds": null}
    ],
    "messages": [...]  # Optional Qwen3VL message format
}

V3 Prompt Format (Referring Task):
----------------------------------
"Action Recognition Task: The first region <|object_ref_start|>person<|object_ref_end|>
<|box_start|>(x1,y1),(x2,y2)<|box_end|> contains a PERSON. The second region 
<|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|> 
contains an OBJECT. The interaction region <|object_ref_start|>interaction<|object_ref_end|>
<|box_start|>(x1,y1),(x2,y2)<|box_end|> shows where the interaction occurs..."
"""

import os
import re
import json
import torch
from PIL import Image
from typing import List, Optional, Tuple

from groma.data.datasets.groma_qwen import (
    GromaInstructQwen,
    BOX_START,
    BOX_END,
    BOX_START_TOKEN_ID,
    BOX_END_TOKEN_ID,
    extract_boxes_from_native_format,
    extract_boxes_from_json_format,
)
from groma.constants import IGNORE_INDEX


def compute_union_box_list(boxes: List[List[int]]) -> List[int]:
    """Compute the union (bounding box) of multiple boxes.
    
    Args:
        boxes: List of [x1, y1, x2, y2] boxes in [0, 1000] range
    
    Returns:
        Union box as [x1, y1, x2, y2]
    """
    if not boxes:
        return [0, 0, 0, 0]
    
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    
    return [x1, y1, x2, y2]


def add_interaction_box_to_prompt(prompt: str, boxes: List[List[int]]) -> Tuple[str, List[List[int]]]:
    """Add interaction box to an existing prompt.
    
    Takes a V2-style prompt with person and object boxes, computes the
    interaction box, and adds it to the prompt.
    
    Args:
        prompt: Original prompt with person and object boxes
        boxes: List of [x1, y1, x2, y2] boxes (person, object)
    
    Returns:
        Tuple of (augmented_prompt, augmented_boxes)
    """
    if len(boxes) < 2:
        # Not enough boxes for interaction
        return prompt, boxes
    
    # Compute interaction box
    person_box = boxes[0]
    object_box = boxes[1]
    interaction_box = compute_union_box_list([person_box, object_box])
    
    # Format interaction box string
    interaction_str = (
        f"<|object_ref_start|>interaction<|object_ref_end|>"
        f"<|box_start|>({interaction_box[0]},{interaction_box[1]}),({interaction_box[2]},{interaction_box[3]})<|box_end|>"
    )
    
    # Find where to insert the interaction box
    # Look for the pattern "contains an OBJECT." and insert after it
    object_pattern = r"contains an OBJECT\."
    match = re.search(object_pattern, prompt)
    
    if match:
        # Insert interaction region description after "contains an OBJECT."
        insert_pos = match.end()
        augmented_prompt = (
            prompt[:insert_pos] + 
            f" The interaction region {interaction_str} shows where the interaction occurs." +
            prompt[insert_pos:]
        )
    else:
        # Fallback: append interaction box at the end before the action request
        action_pattern = r"Describe the action"
        match = re.search(action_pattern, prompt)
        if match:
            insert_pos = match.start()
            augmented_prompt = (
                prompt[:insert_pos] + 
                f"The interaction region {interaction_str} shows the context of the interaction. " +
                prompt[insert_pos:]
            )
        else:
            # Last resort: just append
            augmented_prompt = prompt + f" Interaction region: {interaction_str}"
    
    # Augment boxes
    augmented_boxes = boxes + [interaction_box]
    
    return augmented_prompt, augmented_boxes


class GromaInstructQwenInteraction(GromaInstructQwen):
    """Dataset processor for Groma with Interaction Token.
    
    Extends GromaInstructQwen (V2) to add interaction boxes to prompts.
    
    Key Features:
    - Computes interaction box as union of person + object boxes
    - Adds interaction region to prompt text
    - Creates 3 refer_boxes for ROI extraction
    - Supports toggling interaction on/off for ablation
    """
    
    def __init__(
        self,
        ann_file,
        img_prefix,
        tokenizer,
        img_processor,
        conv_temp='default',
        task_filter=None,
        use_interaction: bool = True,
    ):
        """Initialize GromaInstructQwenInteraction dataset.
        
        Args:
            ann_file: Path to annotation JSON file
            img_prefix: Path prefix for images
            tokenizer: Tokenizer with Qwen3VL tokens
            img_processor: Image processor
            conv_temp: Conversation template (kept for compatibility)
            task_filter: Optional filter - 'referring', 'grounding', or None
            use_interaction: Whether to add interaction boxes (default True)
        """
        super().__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            tokenizer=tokenizer,
            img_processor=img_processor,
            conv_temp=conv_temp,
            task_filter=task_filter,
        )
        
        self.use_interaction = use_interaction
        print(f"[GromaInstructQwenInteraction V3] Initialized with:")
        print(f"  - Use interaction token: {self.use_interaction}")
        print(f"  - Total samples: {len(self.meta_data)}")
    
    def _convert_to_messages_with_interaction(self, data_item):
        """Convert data item to Qwen3VL message format with interaction box.
        
        Extends the parent method to add interaction box to the messages.
        """
        # Get base messages from parent
        messages = self._convert_to_messages(data_item)
        if messages is None:
            return None
        
        if not self.use_interaction:
            return messages
        
        # Only modify referring task prompts
        task_type = data_item.get('task_type', 'referring')
        if task_type != 'referring':
            return messages
        
        # Extract boxes from the user message
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        boxes = extract_boxes_from_native_format(text)
                        
                        if len(boxes) >= 2:
                            # Add interaction box to the text
                            augmented_text, _ = add_interaction_box_to_prompt(text, boxes)
                            item['text'] = augmented_text
                        break
                break
        
        return messages
    
    def preprocess(self, data_item):
        """Preprocess data item with interaction box.
        
        Extends V2 preprocess to add interaction box to refer_boxes.
        """
        img_w = data_item.get('width', 640)
        img_h = data_item.get('height', 480)
        task_type = data_item.get('task_type', 'referring')
        
        # Get messages with interaction box added
        messages = self._convert_to_messages_with_interaction(data_item)
        if messages is None:
            return None
        
        # Extract refer_boxes from user message (including interaction)
        refer_boxes_list = []
        ground_boxes_list = []
        
        conversation = data_item.get('conversation', [])
        for i, conv in enumerate(conversation):
            value = conv.get('value', '')
            
            if i % 2 == 0:  # Human turn
                # Extract boxes from original prompt
                boxes_in_text = extract_boxes_from_native_format(value)
                
                # Add interaction box if enabled and we have enough boxes
                if self.use_interaction and len(boxes_in_text) >= 2:
                    interaction_box = compute_union_box_list(boxes_in_text[:2])
                    boxes_in_text = boxes_in_text + [interaction_box]
                
                refer_boxes_list.extend(boxes_in_text)
            else:  # GPT turn
                # Boxes in GPT turn are ground_boxes (for grounding task)
                boxes_in_text = extract_boxes_from_json_format(value)
                if not boxes_in_text:
                    boxes_in_text = extract_boxes_from_native_format(value)
                ground_boxes_list.extend(boxes_in_text)
        
        # Convert to tensors in [0, 1000] format
        if len(refer_boxes_list) > 0:
            refer_boxes = torch.tensor(refer_boxes_list, dtype=torch.float32)
        else:
            refer_boxes = torch.empty(0, 4)
        
        if len(ground_boxes_list) > 0:
            ground_boxes = torch.tensor(ground_boxes_list, dtype=torch.float32)
        else:
            ground_boxes = torch.empty(0, 4)
        
        # Tokenize using apply_chat_template
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'][0]
        
        # Truncate if needed
        max_seq_length = 2048
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
        
        # Create labels (mask everything except assistant response)
        targets = input_ids.clone()
        
        # Find <|im_start|> positions
        im_start_positions = (input_ids == self.im_start_token_id).nonzero(as_tuple=True)[0]
        
        if len(im_start_positions) >= 3:
            assistant_start = im_start_positions[-1].item()
            targets[:assistant_start] = IGNORE_INDEX
            content_start = min(assistant_start + 5, len(targets))
            targets[assistant_start:content_start] = IGNORE_INDEX
        else:
            targets[:] = IGNORE_INDEX
        
        # Verify we have 3 <|box_end|> tokens for interaction
        num_box_end = (input_ids == self.box_end_token_id).sum().item()
        expected_box_end = 3 if (self.use_interaction and task_type == 'referring') else 2
        
        if num_box_end != expected_box_end and task_type == 'referring':
            # Log mismatch for debugging
            pass  # Could add warning here
        
        return dict(
            input_ids=input_ids,
            labels=targets,
            refer_boxes=refer_boxes,
            ground_boxes=ground_boxes,
            task_type=task_type,
            source='groma_qwen_v3_interaction'
        )


class GromaInstructQwenInteractionV3(GromaInstructQwenInteraction):
    """Alias for explicit version naming."""
    pass


# Utility functions for dataset regeneration

def regenerate_dataset_with_interaction(
    input_json: str,
    output_json: str,
    task_filter: Optional[str] = None,
):
    """Regenerate a dataset JSON file with interaction boxes added to prompts.
    
    This is useful for pre-processing datasets rather than doing it on-the-fly.
    
    Args:
        input_json: Path to input V2 dataset JSON
        output_json: Path to output V3 dataset JSON
        task_filter: Optional filter for task_type
    """
    import json
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    augmented_data = []
    
    for item in data:
        task_type = item.get('task_type', 'referring')
        
        # Skip non-matching tasks
        if task_filter and task_type != task_filter:
            continue
        
        # Only augment referring tasks
        if task_type != 'referring':
            augmented_data.append(item)
            continue
        
        # Augment conversation with interaction box
        conversation = item.get('conversation', [])
        new_conversation = []
        
        for i, conv in enumerate(conversation):
            if i % 2 == 0:  # Human turn
                value = conv.get('value', '')
                boxes = extract_boxes_from_native_format(value)
                
                if len(boxes) >= 2:
                    augmented_value, _ = add_interaction_box_to_prompt(value, boxes)
                    new_conv = dict(conv)
                    new_conv['value'] = augmented_value
                    new_conversation.append(new_conv)
                else:
                    new_conversation.append(conv)
            else:
                new_conversation.append(conv)
        
        # Create augmented item
        new_item = dict(item)
        new_item['conversation'] = new_conversation
        new_item['has_interaction'] = True
        augmented_data.append(new_item)
    
    # Save augmented dataset
    with open(output_json, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"Saved {len(augmented_data)} samples to {output_json}")

