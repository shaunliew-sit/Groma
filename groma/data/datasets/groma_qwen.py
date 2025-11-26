import os
import torch
from PIL import Image
from torchvision import transforms
from groma.data.datasets.groma import GromaInstruct
from groma.constants import DEFAULT_TOKENS, IGNORE_INDEX
from torchvision.ops import box_convert
from groma.data.datasets.det_data import normalize_box_coordinates


class GromaInstructQwen(GromaInstruct):
    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default'):
        """
        Initialize GromaInstructQwen dataset.
        
        Note: conv_temp parameter is kept for compatibility but ignored.
        We use Qwen3VL template format instead.
        """
        # Call parent __init__ but we'll override preprocess
        super().__init__(ann_file, img_prefix, tokenizer, img_processor, conv_temp)
        
        # Verify tokenizer has apply_chat_template method (required for Qwen3VL)
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError(
                "Tokenizer must have apply_chat_template method for Qwen3VL template. "
                "Please ensure you're using Qwen3-VL tokenizer from Groma/checkpoints/Qwen3-VL-8B-Instruct"
            )
        
        # Get EOS token ID (should be 151645 for Qwen3-VL)
        self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if self.eos_token_id is None:
            # Try to find <|im_end|> token
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if self.eos_token_id == self.tokenizer.unk_token_id:
                raise ValueError("Could not find EOS token (<|im_end|>) in tokenizer vocabulary")
        
        # Get <|im_start|> and <|im_end|> token IDs for masking
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        if self.im_start_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Could not find <|im_start|> token in tokenizer vocabulary")
        if self.im_end_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Could not find <|im_end|> token in tokenizer vocabulary")
    
    def _convert_conversation_to_messages(self, conversation, bboxes):
        """
        Convert conversation format to Qwen3VL message format.
        
        Args:
            conversation: List of dicts with "from" and "value" keys
            bboxes: Normalized bounding boxes tensor
        
        Returns:
            List of message dicts in Qwen3VL format
        """
        # System message
        system_message = (
            f"Here is an image with region crops from it. "
            f"Image: {DEFAULT_TOKENS['image']}. "
            f"Regions: {DEFAULT_TOKENS['region']}."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        
        # Process conversation pairs
        assert len(conversation) % 2 == 0, "Conversation must have even number of turns"
        
        for i, conv in enumerate(conversation):
            role = conv.get('from', '').lower()
            value = conv.get('value', '')
            box_inds = conv.get('box_inds', None)
            
            if role == 'human':
                # User message: add [grounding] prefix and handle image placeholder
                text_content = DEFAULT_TOKENS['ground'] + value
                user_content = [
                    {"type": "image", "image": "placeholder"},  # Actual image loaded in __getitem__
                    {"type": "text", "text": text_content}
                ]
                messages.append({
                    "role": "user",
                    "content": user_content
                })
            elif role == 'gpt':
                # Assistant message: preserve Groma tokens and coordinates
                messages.append({
                    "role": "assistant",
                    "content": value
                })
        
        return messages
    
    def preprocess(self, data_item):
        """
        Preprocess data item using Qwen3VL template format.
        
        Uses apply_chat_template() to convert messages to tokenized format,
        then applies proper masking for training (only assistant responses).
        """
        # 1. Load and normalize boxes
        bboxes = data_item['boxes']
        bboxes = torch.tensor(bboxes)
        bboxes = box_convert(bboxes, 'xywh', 'cxcywh')
        img_w = data_item['width']
        img_h = data_item['height']
        bboxes = normalize_box_coordinates(bboxes, (img_h, img_w))
        
        # 2. Get messages (use pre-converted if available, otherwise convert)
        if 'messages' in data_item:
            # Use pre-converted Qwen3VL format from regenerated dataset
            messages = data_item['messages']
        else:
            # Convert from conversation format (backward compatibility)
            conversation = data_item['conversation']
            messages = self._convert_conversation_to_messages(conversation, bboxes)
        
        # 3. Extract boxes for refer_boxes and ground_boxes
        refer_boxes_list = []
        ground_boxes_list = []
        
        # Get conversation for box extraction (use original conversation format)
        conversation = data_item.get('conversation', [])
        for i, conv in enumerate(conversation):
            box_inds = conv.get('box_inds', None)
            if box_inds is not None and len(box_inds) > 0:
                if i % 2 == 0:
                    # Human query: referring boxes
                    refer_boxes_list.extend([bboxes[idx] for idx in box_inds])
                else:
                    # Assistant response: grounding boxes
                    ground_boxes_list.extend([bboxes[idx] for idx in box_inds])
        
        # Stack boxes
        if len(ground_boxes_list) > 0:
            ground_boxes = torch.stack(ground_boxes_list)
        else:
            ground_boxes = torch.empty(0, 4)
        
        if len(refer_boxes_list) > 0:
            refer_boxes = torch.stack(refer_boxes_list)
        else:
            refer_boxes = torch.empty(0, 4)
        
        # 4. Apply Qwen3VL chat template to tokenize
        # Note: We need to replace image placeholder with actual image for tokenization
        # But for now, we'll use a dummy image since the actual image is loaded in __getitem__
        # The template will handle the image token placement
        
        # Create a copy of messages with actual image (will be replaced in __getitem__)
        # For tokenization, we use placeholder - the actual image embedding will be injected later
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'][0]
        
        # Truncate sequences that exceed model's max position embeddings (2048 for Qwen3-VL)
        # Strategy: Preserve system message (contains <image>, <region> tokens) + assistant response (training target)
        # Truncate middle portions (user messages) if needed
        max_seq_length = 2048
        if len(input_ids) > max_seq_length:
            # Find system message end and assistant start
            im_start_positions = (input_ids == self.im_start_token_id).nonzero(as_tuple=True)[0]
            im_end_positions = (input_ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
            
            if len(im_start_positions) >= 3 and len(im_end_positions) >= 2:
                # System message typically ends at first <|im_end|>
                system_end = im_end_positions[0].item() + 1  # Include the <|im_end|> token
                # Assistant starts at last <|im_start|>
                assistant_start = im_start_positions[-1].item()
                
                system_size = system_end
                assistant_size = len(input_ids) - assistant_start
                remaining = max_seq_length - system_size - assistant_size
                
                if remaining >= 0:
                    # Can fit both: system + middle portion + assistant
                    parts = [input_ids[:system_end]]  # System message
                    if remaining > 0:
                        parts.append(input_ids[assistant_start - remaining:assistant_start])  # Middle portion
                    parts.append(input_ids[assistant_start:])  # Assistant response
                    input_ids = torch.cat(parts)
                else:
                    # Can't fit both, prioritize assistant response (training target)
                    # Keep: last portion of system + full assistant
                    system_portion = max(50, max_seq_length - assistant_size)  # Keep at least 50 tokens from system
                    input_ids = torch.cat([
                        input_ids[:system_portion],
                        input_ids[assistant_start:]
                    ])
                    # Final truncation if still too long
                    if len(input_ids) > max_seq_length:
                        input_ids = input_ids[-max_seq_length:]
            else:
                # Fallback: keep last max_seq_length tokens (preserves assistant response)
                input_ids = input_ids[-max_seq_length:]
        
        # 5. Mask targets: only train on assistant responses
        # Qwen3VL format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n
        # We want to mask everything except assistant responses
        
        targets = input_ids.clone()
        
        # Find all <|im_start|> positions
        im_start_positions = (input_ids == self.im_start_token_id).nonzero(as_tuple=True)[0]
        im_end_positions = (input_ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
        
        # Strategy: Find the last <|im_start|> (should be for assistant)
        # Mask everything before it, then find where assistant content actually starts
        # (after "<|im_start|>assistant\n")
        
        if len(im_start_positions) >= 3:  # Should have system, user, assistant
            # Last <|im_start|> should be assistant
            assistant_im_start_idx = im_start_positions[-1].item()
            
            # Mask everything before assistant <|im_start|>
            targets[:assistant_im_start_idx] = IGNORE_INDEX
            
            # Find where assistant content starts (after "<|im_start|>assistant\n")
            # We need to skip: <|im_start|> (1 token) + "assistant" (multiple tokens) + \n (1 token)
            # Approximate: decode a small portion to find where content starts
            # For efficiency, use a fixed offset (typically "assistant" + "\n" is ~3-5 tokens)
            # More robust: search for newline token after assistant_im_start_idx
            content_start = assistant_im_start_idx + 1  # Skip <|im_start|>
            
            # Try to find newline token (token ID 198 in many tokenizers) after assistant role
            # Look ahead up to 10 tokens
            for offset in range(1, min(10, len(input_ids) - assistant_im_start_idx)):
                idx = assistant_im_start_idx + offset
                # Check if this looks like end of role name (newline or space)
                # Token 198 is often newline, but let's be more general
                # Actually, let's just use a reasonable offset
                if offset >= 5:  # Usually "assistant\n" is ~5 tokens
                    content_start = idx + 1
                    break
            
            # Mask the role name part
            if content_start < len(targets):
                targets[assistant_im_start_idx:content_start] = IGNORE_INDEX
            
            # Everything from content_start onwards should be kept (including final <|im_end|>)
        else:
            # Fallback: if structure is unexpected, mask everything before last <|im_end|>
            if len(im_end_positions) >= 2:
                # Keep only the last assistant response (between second-to-last and last <|im_end|>)
                last_eos_before_assistant = im_end_positions[-2].item()
                targets[:last_eos_before_assistant + 1] = IGNORE_INDEX
            else:
                # Very fallback: mask everything
                targets[:] = IGNORE_INDEX
        
        # 6. Return data dict
        data_dict = dict(
            input_ids=input_ids,
            labels=targets,
            ground_boxes=ground_boxes,
            refer_boxes=refer_boxes,
            source='walle_data'
        )
        return data_dict
    
    def __getitem__(self, i) -> dict:
        """
        Get item with image processing.
        Override to handle image in messages for Qwen3VL template.
        """
        data_item = self.meta_data[i]
        image_file = data_item['file_name']
        
        # Load original image (before resize) for region cropping
        original_image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        img_w = data_item['width']
        img_h = data_item['height']
        
        # Resize image for Qwen3VL (448x448)
        image_resized = original_image.resize((448, 448))
        image = self.img_processor.preprocess(image_resized, return_tensors='pt')['pixel_values'][0]
        
        # Preprocess to get tokenized input (includes refer_boxes and ground_boxes)
        data_dict = self.preprocess(data_item)
        data_dict['image'] = image
        
        # Prepare region_images from refer_boxes for DINOv2 processing
        # refer_boxes are in normalized cxcywh format (0-1), need to convert to xyxy pixel coordinates
        refer_boxes = data_dict.get('refer_boxes', torch.empty(0, 4))
        if refer_boxes.numel() > 0:
            # Convert refer_boxes from normalized cxcywh to xyxy pixel coordinates
            # refer_boxes shape: (N, 4) in normalized cxcywh format
            refer_boxes_cxcywh = refer_boxes.clone()
            
            # Denormalize: multiply by image dimensions
            refer_boxes_cxcywh[:, [0, 2]] *= img_w  # cx, w
            refer_boxes_cxcywh[:, [1, 3]] *= img_h  # cy, h
            
            # Convert cxcywh to xyxy
            refer_boxes_xyxy = box_convert(refer_boxes_cxcywh, 'cxcywh', 'xyxy')
            
            # Crop regions from original image
            region_images_list = []
            for bbox_xyxy in refer_boxes_xyxy:
                x1, y1, x2, y2 = bbox_xyxy.tolist()
                # Clip to image bounds
                x1 = max(0, min(int(x1), img_w))
                y1 = max(0, min(int(y1), img_h))
                x2 = max(x1, min(int(x2), img_w))
                y2 = max(y1, min(int(y2), img_h))
                
                # Crop and resize to 224x224 for DINOv2
                if x2 > x1 and y2 > y1:
                    region = original_image.crop((x1, y1, x2, y2))
                    region = region.resize((224, 224), Image.Resampling.LANCZOS)
                    # Convert to tensor (C, H, W) in [0, 1] range
                    transform = transforms.ToTensor()
                    region_tensor = transform(region)
                    region_images_list.append(region_tensor)
            
            if len(region_images_list) > 0:
                # Stack into batch: (N, C, H, W)
                region_images = torch.stack(region_images_list)
            else:
                # No valid regions - return None (matches inference behavior)
                # This ensures training matches inference: both use None when no regions
                region_images = None
        else:
            # No refer boxes - return None (matches inference behavior)
            # For grounding tasks, refer_boxes is empty, so region_images should be None
            # This prevents training/inference mismatch
            region_images = None
        
        data_dict['region_images'] = region_images
        
        return data_dict
