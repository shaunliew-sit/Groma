"""
[GROMA-QWEN V2] Dataset processor for Qwen3VL Format

This module processes datasets in both:
- Native bbox tokens for referring: <|box_start|>, <|box_end|>, <|object_ref_start|>, <|object_ref_end|>
- JSON format for grounding: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]

The JSON format for grounding aligns with Qwen3-VL's pre-trained grounding capabilities.

Dataset Format:
--------------
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

Referring Task Prompt:
    "Action Recognition Task: The first region <|object_ref_start|>person<|object_ref_end|>
    <|box_start|>(x1,y1),(x2,y2)<|box_end|> contains a PERSON..."

Grounding Task Output (JSON format):
    '[{"bbox_2d": [323, 66, 665, 622], "label": "person sitting on bench"}, ...]'
"""

import os
import re
import json
import torch
from PIL import Image
from torchvision import transforms
from groma.data.datasets.groma import GromaInstruct
from groma.constants import IGNORE_INDEX
from torchvision.ops import box_convert

# Native Qwen3VL bbox tokens
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
BOX_START_TOKEN_ID = 151648
BOX_END_TOKEN_ID = 151649


def normalize_box_coordinates(boxes, img_size):
    """Normalize boxes from pixel coords to [0, 1] range.
    
    Args:
        boxes: Tensor of boxes in cxcywh format
        img_size: (height, width) tuple
    
    Returns:
        Normalized boxes in [0, 1] range
    """
    h, w = img_size
    boxes = boxes.clone()
    boxes[:, [0, 2]] /= w  # cx, w
    boxes[:, [1, 3]] /= h  # cy, h
    return boxes


def extract_boxes_from_native_format(text: str) -> list:
    """Extract bounding boxes from native Qwen3VL format.
    
    Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    
    Args:
        text: Text containing native bbox format
    
    Returns:
        List of [x1, y1, x2, y2] boxes in [0, 1000] range
    """
    pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern, text)
    
    boxes = []
    for match in matches:
        x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
        boxes.append([x1, y1, x2, y2])
    
    return boxes


def extract_boxes_from_json_format(text: str) -> list:
    """Extract bounding boxes from JSON format (Qwen3-VL grounding style).
    
    Format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}, ...]
    
    Args:
        text: Text containing JSON bbox format
    
    Returns:
        List of [x1, y1, x2, y2] boxes in [0, 1000] range
    """
    boxes = []
    
    try:
        # Clean markdown fencing if present
        clean_text = text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        data = json.loads(clean_text)
        
        # Handle both single object and array
        if not isinstance(data, list):
            data = [data]
        
        for item in data:
            if "bbox_2d" in item:
                bbox = item["bbox_2d"]
                if len(bbox) == 4:
                    boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # If JSON parsing fails, return empty list
        pass
    
    return boxes


class GromaInstructQwen(GromaInstruct):
    """Dataset processor for Groma with native Qwen3VL format.
    
    Key Features:
    - Uses native Qwen3VL bbox tokens
    - No custom token handling
    - Extracts refer_boxes from native format in prompts
    - Supports task_filter to filter by task type ('referring' or 'grounding')
    """
    
    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default', task_filter=None):
        """Initialize GromaInstructQwen dataset.
        
        Args:
            ann_file: Path to annotation JSON file
            img_prefix: Path prefix for images
            tokenizer: Tokenizer with Qwen3VL tokens
            img_processor: Image processor
            conv_temp: Conversation template (kept for compatibility, ignored)
            task_filter: Optional filter - 'referring', 'grounding', or None for all
        """
        super().__init__(ann_file, img_prefix, tokenizer, img_processor, conv_temp)
        
        # Apply task filter if specified
        self.task_filter = task_filter
        if task_filter is not None:
            original_count = len(self.meta_data)
            self.meta_data = [
                item for item in self.meta_data 
                if item.get('task_type', 'referring') == task_filter
            ]
            filtered_count = len(self.meta_data)
            print(f"[GromaInstructQwen V2] Task filter '{task_filter}': {original_count} -> {filtered_count} samples")
        
        # Verify tokenizer has apply_chat_template method
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError(
                "Tokenizer must have apply_chat_template method for Qwen3VL template."
            )
        
        # Get native Qwen3VL token IDs
        self.box_start_token_id = self.tokenizer.convert_tokens_to_ids(BOX_START)
        self.box_end_token_id = self.tokenizer.convert_tokens_to_ids(BOX_END)
        
        if self.box_start_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Native token {BOX_START} not found in tokenizer")
        if self.box_end_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Native token {BOX_END} not found in tokenizer")
        
        # Get EOS token ID
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_id = self.im_end_token_id
        
        print(f"[GromaInstructQwen V2] Initialized with native Qwen3VL tokens:")
        print(f"  - {BOX_START}: {self.box_start_token_id}")
        print(f"  - {BOX_END}: {self.box_end_token_id}")
        print(f"  - Total samples: {len(self.meta_data)}")
    
    def _convert_to_messages(self, data_item):
        """Convert data item to Qwen3VL message format.
        
        Uses pre-converted messages if available, otherwise converts conversation.
        """
        if 'messages' in data_item and data_item['messages']:
            return data_item['messages']
        
        conversation = data_item.get('conversation', [])
        if len(conversation) < 2:
            return None
        
        task_type = data_item.get('task_type', 'referring')
        
        # System message based on task type
        if task_type == 'referring':
            system_content = (
                "You are an expert at understanding human-object interactions in images. "
                "You will be given an image with two regions marked by bounding boxes. "
                "Describe the action being performed."
            )
        else:
            # JSON format grounding system prompt (aligned with Qwen3-VL)
            # Uses simplified labels to avoid confusion between person and object entries
            system_content = (
                "You are an expert at locating human-object interactions in images. "
                "When asked to locate people performing actions with objects, find ALL matching "
                "person-object pairs and output their bounding box coordinates in JSON format. "
                "Use bbox_2d as [x1, y1, x2, y2] coordinates (0-1000 scale). "
                "Label person boxes as 'person' and object boxes with the object category name. "
                "Output pairs in alternating order: [person, object, person, object, ...]."
            )
        
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        for i, conv in enumerate(conversation):
            role = conv.get('from', '').lower()
            value = conv.get('value', '')
            
            if role == 'human':
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": value}
                    ]
                })
            elif role == 'gpt':
                messages.append({
                    "role": "assistant",
                    "content": value
                })
        
        return messages
    
    def preprocess(self, data_item):
        """Preprocess data item using native Qwen3VL format.
        
        Returns:
            dict with input_ids, labels, refer_boxes, ground_boxes
        """
        img_w = data_item.get('width', 640)
        img_h = data_item.get('height', 480)
        task_type = data_item.get('task_type', 'referring')
        
        # Get messages
        messages = self._convert_to_messages(data_item)
        if messages is None:
            return None
        
        # Extract refer_boxes from user message (for referring task)
        refer_boxes_list = []
        ground_boxes_list = []
        
        conversation = data_item.get('conversation', [])
        for i, conv in enumerate(conversation):
            value = conv.get('value', '')
            
            if i % 2 == 0:  # Human turn
                # Boxes in human turn are refer_boxes (for referring task)
                # Always use native format for human prompts (referring)
                boxes_in_text = extract_boxes_from_native_format(value)
                refer_boxes_list.extend(boxes_in_text)
            else:  # GPT turn
                # Boxes in GPT turn are ground_boxes (for grounding task)
                # Try JSON format first (for grounding), fall back to native format
                boxes_in_text = extract_boxes_from_json_format(value)
                if not boxes_in_text:
                    # Fall back to native format for backward compatibility
                    boxes_in_text = extract_boxes_from_native_format(value)
                ground_boxes_list.extend(boxes_in_text)
        
        # Convert to tensors in [0, 1000] format (already in this format from native)
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
            # Keep last portion to preserve assistant response
            input_ids = input_ids[-max_seq_length:]
        
        # Create labels (mask everything except assistant response)
        targets = input_ids.clone()
        
        # Find <|im_start|> positions
        im_start_positions = (input_ids == self.im_start_token_id).nonzero(as_tuple=True)[0]
        
        if len(im_start_positions) >= 3:
            # Last <|im_start|> should be assistant
            assistant_start = im_start_positions[-1].item()
            
            # Mask everything before assistant
            targets[:assistant_start] = IGNORE_INDEX
            
            # Also mask the role part (assistant\n) - approximately 5 tokens
            content_start = min(assistant_start + 5, len(targets))
            targets[assistant_start:content_start] = IGNORE_INDEX
        else:
            # Fallback: mask everything
            targets[:] = IGNORE_INDEX
        
        return dict(
            input_ids=input_ids,
            labels=targets,
            refer_boxes=refer_boxes,
            ground_boxes=ground_boxes,
            task_type=task_type,
            source='groma_qwen_v2'
        )
    
    def __getitem__(self, i) -> dict:
        """Get item with image processing.
        
        Returns dict with:
        - input_ids: Tokenized input
        - labels: Training labels
        - image: Processed image for Qwen3VL
        - refer_boxes: Boxes for referring task (in [0, 1000] format)
        - ground_boxes: Boxes for grounding task output
        """
        data_item = self.meta_data[i]
        image_file = data_item['file_name']
        img_w = data_item.get('width', 640)
        img_h = data_item.get('height', 480)
        
        # Load original image
        original_image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        # Resize for Qwen3VL (448x448)
        image_resized = original_image.resize((448, 448))
        image = self.img_processor.preprocess(image_resized, return_tensors='pt')['pixel_values'][0]
        
        # Preprocess to get tokenized input
        data_dict = self.preprocess(data_item)
        if data_dict is None:
            # Return empty dict for invalid samples
            return {
                'input_ids': torch.tensor([]),
                'labels': torch.tensor([]),
                'image': image,
                'refer_boxes': torch.empty(0, 4),
                'ground_boxes': torch.empty(0, 4),
            }
        
        data_dict['image'] = image
        
        # Note: In V2, we don't create region_images here
        # The model extracts features from Qwen3VL vision encoder directly
        # refer_boxes are already in [0, 1000] format from the native format
        
        return data_dict


class GromaInstructQwenNative(GromaInstructQwen):
    """Alias for backward compatibility."""
    pass
