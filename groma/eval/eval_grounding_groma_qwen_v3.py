"""
HICO-DET / SWIG-HOI Grounding Evaluation Script for Groma Qwen V2/V3 (JSON Format)

Evaluates Groma Qwen V2 or V3 (with interaction token) grounding performance 
using JSON bbox format aligned with Qwen3-VL.

Key Features:
- Auto-detects V2 vs V3 models from config.json
- Uses JSON format for grounding: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
- Aligns with Qwen3-VL's pre-trained grounding capabilities
- Supports multi-pair detection in single image

Note: For grounding tasks, there are no input bounding boxes (the model predicts them),
so the interaction token feature doesn't apply. V3 models work the same as V2 for grounding.

Task: Given "Locate every person who is {action} {object} and the {object} they interact with...",
      predict bounding boxes for ALL person-object pairs performing that action in JSON format.

Metrics: Pair-level Precision, Recall, F1 @ IoU thresholds (0.5 to 0.95)
Size-specific metrics: ARs (small), ARm (medium), ARl (large)
"""

import os
import json
import re
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from groma.model.groma_qwen import GromaQwenModel, GromaQwenConfig
from groma.model.groma_qwen_interaction import GromaQwenInteractionModel, GromaQwenInteractionConfig
from groma.utils import disable_torch_init

# Register custom models
AutoConfig.register("groma_qwen", GromaQwenConfig)
AutoModelForCausalLM.register(GromaQwenConfig, GromaQwenModel)
AutoConfig.register("groma_qwen_interaction", GromaQwenInteractionConfig)
AutoModelForCausalLM.register(GromaQwenInteractionConfig, GromaQwenInteractionModel)

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Native Qwen3VL bbox tokens
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
OBJECT_REF_START = "<|object_ref_start|>"
OBJECT_REF_END = "<|object_ref_end|>"


def visualize_grounding_result(
    image: Image.Image,
    gt_pairs: list,
    pred_pairs: list,
    action: str,
    object_category: str,
    output_path: str,
    matched_indices: set = None
):
    """
    Visualize grounding results with GT and predicted boxes.
    
    Args:
        image: PIL Image
        gt_pairs: List of GT pairs, each pair is {'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2]}
        pred_pairs: List of predicted pairs (same format)
        action: Action name
        object_category: Object category name
        output_path: Where to save the visualization
        matched_indices: Set of GT indices that were matched (for color coding)
    """
    # Create a copy for drawing
    viz_img = image.copy()
    draw = ImageDraw.Draw(viz_img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11)
        except:
            font = ImageFont.load_default()
            font_small = font
    
    img_w, img_h = image.size
    matched_indices = matched_indices or set()
    
    # Colors
    GT_PERSON_COLOR = "#00AA00"  # Green for GT person
    GT_OBJECT_COLOR = "#00CC00"  # Light green for GT object
    GT_MATCHED_PERSON = "#0066CC"  # Blue for matched GT
    GT_MATCHED_OBJECT = "#0099FF"  # Light blue for matched GT
    PRED_PERSON_COLOR = "#FF4444"  # Red for predicted person
    PRED_OBJECT_COLOR = "#FF8800"  # Orange for predicted object
    
    # Draw GT pairs first (in background)
    for i, gt_pair in enumerate(gt_pairs):
        person_box = gt_pair['person_box']
        object_box = gt_pair['object_box']
        
        # Choose color based on whether it was matched
        if i in matched_indices:
            person_color = GT_MATCHED_PERSON
            object_color = GT_MATCHED_OBJECT
            label_prefix = f"GT{i+1}✓"
        else:
            person_color = GT_PERSON_COLOR
            object_color = GT_OBJECT_COLOR
            label_prefix = f"GT{i+1}"
        
        # Draw person box
        draw.rectangle(person_box, outline=person_color, width=3)
        # Draw label
        label = f"{label_prefix}-P"
        try:
            bbox = draw.textbbox((person_box[0], max(0, person_box[1] - 18)), label, font=font_small)
            draw.rectangle(bbox, fill=person_color)
            draw.text((person_box[0], max(0, person_box[1] - 18)), label, fill="white", font=font_small)
        except:
            draw.text((person_box[0], max(0, person_box[1] - 15)), label, fill=person_color, font=font_small)
        
        # Draw object box
        draw.rectangle(object_box, outline=object_color, width=3)
        # Draw label
        label = f"{label_prefix}-O"
        try:
            bbox = draw.textbbox((object_box[0], max(0, object_box[1] - 18)), label, font=font_small)
            draw.rectangle(bbox, fill=object_color)
            draw.text((object_box[0], max(0, object_box[1] - 18)), label, fill="white", font=font_small)
        except:
            draw.text((object_box[0], max(0, object_box[1] - 15)), label, fill=object_color, font=font_small)
        
        # Draw connection line
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        draw.line([person_center, object_center], fill=person_color, width=2)
    
    # Draw predicted pairs on top
    for i, pred_pair in enumerate(pred_pairs):
        person_box = pred_pair['person_box']
        object_box = pred_pair['object_box']
        
        # Draw person box (dashed effect using multiple rectangles)
        draw.rectangle(person_box, outline=PRED_PERSON_COLOR, width=2)
        # Inner rectangle for double-line effect
        inner_box = [person_box[0]+2, person_box[1]+2, person_box[2]-2, person_box[3]-2]
        if inner_box[2] > inner_box[0] and inner_box[3] > inner_box[1]:
            draw.rectangle(inner_box, outline=PRED_PERSON_COLOR, width=1)
        
        # Draw label
        label = f"Pred{i+1}-P"
        label_y = min(img_h - 20, person_box[3] + 2)
        try:
            bbox = draw.textbbox((person_box[0], label_y), label, font=font_small)
            draw.rectangle(bbox, fill=PRED_PERSON_COLOR)
            draw.text((person_box[0], label_y), label, fill="white", font=font_small)
        except:
            draw.text((person_box[0], label_y), label, fill=PRED_PERSON_COLOR, font=font_small)
        
        # Draw object box
        draw.rectangle(object_box, outline=PRED_OBJECT_COLOR, width=2)
        inner_box = [object_box[0]+2, object_box[1]+2, object_box[2]-2, object_box[3]-2]
        if inner_box[2] > inner_box[0] and inner_box[3] > inner_box[1]:
            draw.rectangle(inner_box, outline=PRED_OBJECT_COLOR, width=1)
        
        # Draw label
        label = f"Pred{i+1}-O"
        label_y = min(img_h - 20, object_box[3] + 2)
        try:
            bbox = draw.textbbox((object_box[0], label_y), label, font=font_small)
            draw.rectangle(bbox, fill=PRED_OBJECT_COLOR)
            draw.text((object_box[0], label_y), label, fill="white", font=font_small)
        except:
            draw.text((object_box[0], label_y), label, fill=PRED_OBJECT_COLOR, font=font_small)
        
        # Draw connection line
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        draw.line([person_center, object_center], fill=PRED_PERSON_COLOR, width=2)
    
    # Add title/legend at the top
    title = f"Action: {action} | Object: {object_category}"
    legend = f"GT pairs: {len(gt_pairs)} (green/blue=matched) | Pred pairs: {len(pred_pairs)} (red/orange)"
    
    # Draw title background
    try:
        title_bbox = draw.textbbox((10, 5), title, font=font)
        draw.rectangle([5, 2, title_bbox[2] + 5, title_bbox[3] + 3], fill="white", outline="black")
        draw.text((10, 5), title, fill="black", font=font)
        
        legend_bbox = draw.textbbox((10, title_bbox[3] + 5), legend, font=font_small)
        draw.rectangle([5, title_bbox[3] + 2, legend_bbox[2] + 5, legend_bbox[3] + 3], fill="white", outline="black")
        draw.text((10, title_bbox[3] + 5), legend, fill="black", font=font_small)
    except:
        draw.text((10, 5), title, fill="black", font=font)
        draw.text((10, 25), legend, fill="black", font=font_small)
    
    # Save visualization
    viz_img.save(output_path, "JPEG", quality=90)


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def get_box_area(box):
    """Calculate area of bounding box [x1, y1, x2, y2]"""
    return (box[2] - box[0]) * (box[3] - box[1])


def categorize_pair_by_size(gt_pair, area_small=1024, area_medium=9216):
    """
    Categorize a ground truth pair by object size.
    Uses the object box area for categorization (COCO standard).

    Returns: 'small', 'medium', or 'large'
    """
    object_area = get_box_area(gt_pair['object_box'])

    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    else:
        return 'large'


def parse_grounding_response(response_text, img_shape):
    """
    Parse grounding response to extract person-object pairs.
    Supports both JSON format and legacy native Qwen3VL bbox format.

    Args:
        response_text: Generated text in JSON format like:
            '[{"bbox_2d": [320, 306, 359, 349], "label": "person sitting on bench"}, 
              {"bbox_2d": [148, 345, 376, 414], "label": "bench"}]'
            Or legacy native format:
            "<|object_ref_start|>person<|object_ref_end|><|box_start|>(320,306),(359,349)<|box_end|>..."
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs: [{'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2]}, ...]
        Boxes are in pixel coordinates
    """
    pairs = []
    h, w = img_shape

    # Try JSON format first (primary format)
    json_pairs = _parse_json_format(response_text, h, w)
    if json_pairs:
        return json_pairs
    
    # Fall back to native format for backward compatibility
    return _parse_native_format(response_text, h, w)


def _parse_json_format(response_text, h, w):
    """Parse JSON format grounding response."""
    pairs = []
    
    try:
        # Clean markdown fencing if present
        clean_text = response_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            # Handle case where it's just ```...```
            parts = clean_text.split("```")
            if len(parts) >= 2:
                clean_text = parts[1].strip()
        
        # Try to find JSON array in the text
        # Sometimes the model outputs text before/after the JSON
        json_start = clean_text.find('[')
        json_end = clean_text.rfind(']')
        if json_start != -1 and json_end != -1:
            clean_text = clean_text[json_start:json_end + 1]
        
        # Parse JSON
        data = json.loads(clean_text)
        
        # Handle both single object and array
        if not isinstance(data, list):
            data = [data]
        
        if len(data) == 0:
            return []
        
        # Group consecutive pairs: [person1, object1, person2, object2, ...]
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                person_item = data[i]
                object_item = data[i + 1]
                
                if "bbox_2d" in person_item and "bbox_2d" in object_item:
                    person_bbox = person_item["bbox_2d"]
                    object_bbox = object_item["bbox_2d"]
                    
                    # Convert from [0, 1000] to pixel coordinates
                    person_box = [
                        (person_bbox[0] / 1000.0) * w,
                        (person_bbox[1] / 1000.0) * h,
                        (person_bbox[2] / 1000.0) * w,
                        (person_bbox[3] / 1000.0) * h
                    ]
                    
                    object_box = [
                        (object_bbox[0] / 1000.0) * w,
                        (object_bbox[1] / 1000.0) * h,
                        (object_bbox[2] / 1000.0) * w,
                        (object_bbox[3] / 1000.0) * h
                    ]
                    
                    pairs.append({
                        'person_box': person_box,
                        'object_box': object_box
                    })
        
        return pairs
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        return []


def _parse_native_format(response_text, h, w):
    """Parse legacy native Qwen3VL bbox format (fallback)."""
    pairs = []
    
    # Pattern for native Qwen3VL format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    box_pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    
    # Find all box matches
    box_matches = list(re.finditer(box_pattern, response_text))
    
    # Group boxes into pairs (person, object)
    for i in range(0, len(box_matches), 2):
        if i + 1 < len(box_matches):
            # First match is person, second is object
            person_match = box_matches[i]
            object_match = box_matches[i + 1]
            
            # Extract coordinates (in [0, 1000] format from Qwen3VL)
            person_coords = person_match.groups()
            object_coords = object_match.groups()
            
            # Convert from [0, 1000] to pixel coordinates
            person_box = [
                (int(person_coords[0]) / 1000.0) * w,
                (int(person_coords[1]) / 1000.0) * h,
                (int(person_coords[2]) / 1000.0) * w,
                (int(person_coords[3]) / 1000.0) * h
            ]
            
            object_box = [
                (int(object_coords[0]) / 1000.0) * w,
                (int(object_coords[1]) / 1000.0) * h,
                (int(object_coords[2]) / 1000.0) * w,
                (int(object_coords[3]) / 1000.0) * h
            ]
            
            pairs.append({
                'person_box': person_box,
                'object_box': object_box
            })

    return pairs


def match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5):
    """
    Match predicted pairs to ground truth pairs using greedy matching.

    A pair matches if BOTH person and object boxes have IoU > threshold with GT.

    Args:
        pred_pairs: List of predicted pairs (each has 'person_box' and 'object_box')
        gt_pairs: List of ground truth pairs (same format)
        iou_threshold: IoU threshold for matching

    Returns:
        matches: List of (pred_idx, gt_idx, person_iou, object_iou)
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched GT indices
    """
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Build IoU matrix
    iou_matrix = []
    for pred_pair in pred_pairs:
        row = []
        for gt_pair in gt_pairs:
            person_iou = calculate_iou(pred_pair['person_box'], gt_pair['person_box'])
            object_iou = calculate_iou(pred_pair['object_box'], gt_pair['object_box'])

            # Both boxes must match
            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                # Use average IoU as score
                avg_iou = (person_iou + object_iou) / 2.0
                row.append(avg_iou)
            else:
                row.append(0.0)
        iou_matrix.append(row)

    # Greedy matching: pick best match iteratively
    while True:
        best_score = 0.0
        best_pred_idx = -1
        best_gt_idx = -1

        for pred_idx in range(len(pred_pairs)):
            if pred_idx in matched_preds:
                continue
            for gt_idx in range(len(gt_pairs)):
                if gt_idx in matched_gts:
                    continue
                if iou_matrix[pred_idx][gt_idx] > best_score:
                    best_score = iou_matrix[pred_idx][gt_idx]
                    best_pred_idx = pred_idx
                    best_gt_idx = gt_idx

        if best_score == 0.0:
            break

        # Add match
        pred_pair = pred_pairs[best_pred_idx]
        gt_pair = gt_pairs[best_gt_idx]
        person_iou = calculate_iou(pred_pair['person_box'], gt_pair['person_box'])
        object_iou = calculate_iou(pred_pair['object_box'], gt_pair['object_box'])

        matches.append((best_pred_idx, best_gt_idx, person_iou, object_iou))
        matched_preds.add(best_pred_idx)
        matched_gts.add(best_gt_idx)

    unmatched_preds = [i for i in range(len(pred_pairs)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_pairs)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def load_groma_qwen_model(model_name, base_model_name=None, gpu_id=0):
    """
    Load Groma Qwen model with auto-detection for V2 vs V3.
    
    Returns:
        model: GromaQwenModel (V2) or GromaQwenInteractionModel (V3)
        processor: Qwen processor
        tokenizer: Tokenizer
        is_v3: Boolean indicating if this is a V3 model
    """
    model_name = os.path.expanduser(model_name)
    
    print("=" * 80)
    print(f"[GROMA-QWEN V3] Loading model from {model_name}")
    print("=" * 80)

    # Auto-detect model type from config.json
    config_path = os.path.join(model_name, "config.json")
    is_v3 = False
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get("model_type", "groma_qwen")
        is_v3 = model_type == "groma_qwen_interaction"
        print(f"✓ Detected model_type: {model_type}")
    else:
        print(f"! config.json not found, assuming V2 model")
    
    # Load appropriate config and model class
    if is_v3:
        print(f"✓ Loading as V3 (Interaction Token) model")
        config = GromaQwenInteractionConfig.from_pretrained(model_name, trust_remote_code=True)
        model_class = GromaQwenInteractionModel
    else:
        print(f"✓ Loading as V2 (Standard) model")
        config = GromaQwenConfig.from_pretrained(model_name, trust_remote_code=True)
        model_class = GromaQwenModel
    
    print(f"✓ Config loaded (model_type: {config.model_type})")

    # Load Processor
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Processor loaded from checkpoint")
    except Exception as e:
        print(f"! Could not load from checkpoint: {e}")
        if base_model_name:
            processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
            print(f"✓ Processor loaded from base model")
        else:
            raise

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Verify native bbox tokens
    box_start_id = tokenizer.convert_tokens_to_ids(BOX_START)
    box_end_id = tokenizer.convert_tokens_to_ids(BOX_END)
    print(f"✓ Native bbox tokens verified:")
    print(f"  - {BOX_START}: {box_start_id}")
    print(f"  - {BOX_END}: {box_end_id}")

    # Load Model
    disable_torch_init()
    
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        device_map = {"": device}
        dtype = torch.bfloat16
        print(f"✓ Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        device_map = {"": device}
        dtype = torch.float32
        print("! Using CPU")
    
    model = model_class.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    model.init_special_token_id(tokenizer)
    model.eval()
    
    version_str = "V3 (Interaction Token)" if is_v3 else "V2 (Standard)"
    print(f"✓ Model loaded successfully as {version_str}")
    
    return model, processor, tokenizer, is_v3


def build_grounding_prompt(action, object_category):
    """
    Build prompt for grounding task using JSON format (Qwen3-VL aligned).
    
    Note: Grounding task has no input bounding boxes - the model predicts them.
    
    Args:
        action: Action verb (e.g., "sitting on")
        object_category: Object category (e.g., "bench")
    
    Returns:
        Prompt text string requesting JSON format output
    """
    prompt_text = (
        f"Locate every person who is {action} {object_category} and the {object_category} "
        f"they interact with in this image. For each person-object pair, output bbox "
        f"coordinates in JSON format like: {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}"
    )
    return prompt_text


def run_groma_qwen_inference(model, processor, image_path, action, object_category, is_v3=False, gpu_id=0):
    """
    Run Groma Qwen inference for grounding task.
    
    Note: For grounding tasks, there are no input bounding boxes.
    The model predicts the bounding boxes directly.
    V2 and V3 models work the same for grounding.

    Args:
        model: GromaQwenModel (V2) or GromaQwenInteractionModel (V3)
        processor: Processor
        image_path: Path to image file
        action: Action verb
        object_category: Object category name
        is_v3: Whether this is a V3 model (unused for grounding, kept for API consistency)
        gpu_id: GPU device ID

    Returns:
        output_text: Generated response text
        image: PIL Image object
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Build prompt (same for V2 and V3 - no input boxes for grounding)
    prompt_text = build_grounding_prompt(action, object_category)

    # Create messages for Qwen3-VL chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to model device
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    # Get EOS token ID
    tokenizer = processor if hasattr(processor, 'vocab_size') else processor.tokenizer
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id == tokenizer.unk_token_id:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    # Generate - no region_images needed for grounding (model generates boxes)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,  # Allow for multiple pairs
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False,
        )

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return output_text, image


def eval_model(args):
    """Main evaluation function"""

    # Determine dataset name from annotation file
    dataset_name = "HICO-DET" if "hico" in args.ann_file.lower() else "SWIG-HOI"

    print("=" * 80)
    print(f"[GROMA-QWEN V3] {dataset_name} Grounding Evaluation")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Base Model:  {args.base_model_name}")
    print(f"Annotation:  {args.ann_file}")
    print(f"Images:      {args.img_prefix}")
    print(f"Output:      {args.result_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    print("=" * 80)
    print()

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse GPU ID from device string or use default
    gpu_id = 0
    if args.device.startswith("cuda:"):
        gpu_id = int(args.device.split(":")[1])
    elif args.device.isdigit():
        gpu_id = int(args.device)

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"groma_qwen_v3_ground_{dataset_name.lower()}_{timestamp}",
                config={
                    "model": args.model_name,
                    "base_model": args.base_model_name,
                    "dataset": dataset_name,
                    "task": "grounding",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["groma-qwen-v3", dataset_name.lower(), "grounding"]
            )
            print(f"✓ Weights & Biases initialized successfully!")
            print(f"  Run URL: {wandb.run.url}\n")
        except Exception as e:
            print(f"! Warning: WandB initialization failed: {e}")
            print(f"  Continuing evaluation without WandB logging...\n")
            use_wandb = False

    # Load Groma Qwen model with auto-detection
    model, processor, tokenizer, is_v3 = load_groma_qwen_model(
        args.model_name,
        args.base_model_name,
        gpu_id
    )
    
    version_str = "V3 (Interaction Token)" if is_v3 else "V2 (Standard)"
    print(f"\n✓ Model loaded as: {version_str}")
    print("  Note: Grounding task does not use interaction token (no input boxes)")

    # Load annotation file
    print(f"\nLoading annotations from: {args.ann_file}")
    with open(args.ann_file, 'r') as f:
        dataset_samples = json.load(f)

    print(f"Loaded {len(dataset_samples)} samples")

    # Limit dataset if requested
    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"\n⚠️  Limiting evaluation to first {args.max_images} samples")
        dataset_samples = dataset_samples[:args.max_images]

    print(f"\nDataset: {len(dataset_samples)} samples")
    print(f"Each sample = one (action, object) combination")
    print("=" * 80)

    # Evaluation metrics at different IoU thresholds
    iou_thresholds_ar = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # [0.5, 0.55, ..., 0.95]

    # COCO area thresholds for small/medium/large objects
    AREA_SMALL = 32 ** 2    # < 1024 pixels²
    AREA_MEDIUM = 96 ** 2   # < 9216 pixels²

    results_per_threshold = {
        iou_thr: {
            'tp': 0,  # True positives (matched pairs)
            'fp': 0,  # False positives (unmatched predictions)
            'fn': 0,  # False negatives (unmatched GT pairs)
            'tp_small': 0,
            'fn_small': 0,
            'tp_medium': 0,
            'fn_medium': 0,
            'tp_large': 0,
            'fn_large': 0,
        }
        for iou_thr in iou_thresholds_ar
    }

    # Per-sample results
    per_sample_results = []

    # Per-action statistics
    action_stats = defaultdict(lambda: {
        'total_samples': 0,
        'total_gt_pairs': 0,
        'total_pred_pairs': 0,
        'matched_pairs_05': 0
    })

    show_verbose = args.verbose or len(dataset_samples) <= 100

    # Create visualization directory if verbose mode
    viz_dir = None
    if show_verbose:
        viz_dir = args.result_file.replace('.json', '_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}\n")

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        file_name = sample['file_name']
        action = sample['action']
        object_category = sample['object_category']
        img_path = os.path.join(args.img_prefix, file_name)
        img_shape = (sample['height'], sample['width'])

        # Build GT pairs from sample boxes
        gt_boxes_list = sample['boxes']
        gt_pairs = []

        # Verify we have an even number of boxes (pairs)
        if len(gt_boxes_list) % 2 != 0:
            print(f"WARNING: Sample {file_name} has odd number of boxes ({len(gt_boxes_list)})")

        for i in range(0, len(gt_boxes_list), 2):
            if i + 1 < len(gt_boxes_list):
                person_box = gt_boxes_list[i]
                object_box = gt_boxes_list[i + 1]

                # Verify boxes are in correct format [x1, y1, x2, y2]
                if not (isinstance(person_box, list) and len(person_box) == 4):
                    print(f"WARNING: Invalid person box format at {file_name}, pair {i//2}: {person_box}")
                    continue
                if not (isinstance(object_box, list) and len(object_box) == 4):
                    print(f"WARNING: Invalid object box format at {file_name}, pair {i//2}: {object_box}")
                    continue

                gt_pairs.append({
                    'person_box': person_box,
                    'object_box': object_box
                })

        if show_verbose:
            print(f"\n[Sample {idx+1}/{len(dataset_samples)}] {file_name}")
            print(f"  Action: {action}, Object: {object_category}")
            print(f"  GT boxes loaded: {len(gt_boxes_list)}, GT pairs: {len(gt_pairs)}")

        # Build prompt
        prompt_text = build_grounding_prompt(action, object_category)

        # Run Groma Qwen inference
        output_text, image = run_groma_qwen_inference(
            model, processor, img_path, action, object_category, is_v3, gpu_id
        )

        if show_verbose:
            print(f"  Prompt: {prompt_text}")
            print(f"  Response: {output_text[:200]}...")

        # Parse predicted pairs
        pred_pairs = parse_grounding_response(output_text, img_shape)

        if show_verbose:
            print(f"  Predicted pairs: {len(pred_pairs)}")

        # Update action stats
        action_stats[action]['total_samples'] += 1
        action_stats[action]['total_gt_pairs'] += len(gt_pairs)
        action_stats[action]['total_pred_pairs'] += len(pred_pairs)

        # Match predictions to GT at different IoU thresholds
        sample_result = {
            'file_name': file_name,
            'action': action,
            'object': object_category,
            'action_object_id': sample.get('action_object_id', f"{action}_{object_category}"),
            'num_gt_pairs': len(gt_pairs),
            'num_pred_pairs': len(pred_pairs),
            'prompt': prompt_text,
            'generated_text': output_text,
            'matches_per_threshold': {},
            'model_version': 'v3' if is_v3 else 'v2'
        }

        matched_gt_indices_05 = set()  # Initialize for visualization
        
        for iou_thr in iou_thresholds_ar:
            matches, unmatched_preds, unmatched_gts = match_pairs_greedy(
                pred_pairs, gt_pairs, iou_threshold=iou_thr
            )

            # Update overall metrics
            results_per_threshold[iou_thr]['tp'] += len(matches)
            results_per_threshold[iou_thr]['fp'] += len(unmatched_preds)
            results_per_threshold[iou_thr]['fn'] += len(unmatched_gts)

            # Update size-specific metrics
            matched_gt_indices = {m[1] for m in matches}
            for gt_idx, gt_pair in enumerate(gt_pairs):
                size_category = categorize_pair_by_size(gt_pair, AREA_SMALL, AREA_MEDIUM)
                if gt_idx in matched_gt_indices:
                    # This GT was matched (True Positive for this size category)
                    results_per_threshold[iou_thr][f'tp_{size_category}'] += 1
                else:
                    # This GT was not matched (False Negative for this size category)
                    results_per_threshold[iou_thr][f'fn_{size_category}'] += 1

            sample_result['matches_per_threshold'][iou_thr] = {
                'matched': len(matches),
                'unmatched_preds': len(unmatched_preds),
                'unmatched_gts': len(unmatched_gts)
            }

            if iou_thr == 0.5:
                action_stats[action]['matched_pairs_05'] += len(matches)
                matched_gt_indices_05 = matched_gt_indices  # Save for visualization

                if show_verbose:
                    print(f"  Matched @ IoU=0.5: {len(matches)}/{len(gt_pairs)}")

        per_sample_results.append(sample_result)
        
        # Save visualization if verbose mode
        if show_verbose and viz_dir:
            try:
                viz_filename = f"{idx+1:04d}_{os.path.splitext(file_name)[0]}_{action.replace(' ', '_')}_{object_category.replace(' ', '_')}.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                visualize_grounding_result(
                    image=image,
                    gt_pairs=gt_pairs,
                    pred_pairs=pred_pairs,
                    action=action,
                    object_category=object_category,
                    output_path=viz_path,
                    matched_indices=matched_gt_indices_05
                )
            except Exception as e:
                print(f"  Warning: Could not save visualization: {e}")

        # Log per-sample metrics to WandB
        if use_wandb:
            matches_05 = sample_result['matches_per_threshold'][0.5]['matched']
            recall_05 = matches_05 / len(gt_pairs) if len(gt_pairs) > 0 else 0.0
            wandb.log({
                'sample_idx': idx,
                'recall@0.5': recall_05,
                'num_pred_pairs': len(pred_pairs),
                'num_gt_pairs': len(gt_pairs),
                'num_matched@0.5': matches_05,
            })

    # Compute Average Recall (AR) metrics
    print("\n" + "=" * 80)
    print(f"[GROMA-QWEN V3] {dataset_name} Grounding Evaluation Results")
    print(f"Model Version: {version_str}")
    print("=" * 80)

    # Compute recalls at all IoU thresholds
    recalls = []
    recalls_small = []
    recalls_medium = []
    recalls_large = []

    for iou_thr in iou_thresholds_ar:
        # Overall recall
        tp = results_per_threshold[iou_thr]['tp']
        fn = results_per_threshold[iou_thr]['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

        # Small object recall
        tp_small = results_per_threshold[iou_thr]['tp_small']
        fn_small = results_per_threshold[iou_thr]['fn_small']
        recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        recalls_small.append(recall_small)

        # Medium object recall
        tp_medium = results_per_threshold[iou_thr]['tp_medium']
        fn_medium = results_per_threshold[iou_thr]['fn_medium']
        recall_medium = tp_medium / (tp_medium + fn_medium) if (tp_medium + fn_medium) > 0 else 0.0
        recalls_medium.append(recall_medium)

        # Large object recall
        tp_large = results_per_threshold[iou_thr]['tp_large']
        fn_large = results_per_threshold[iou_thr]['fn_large']
        recall_large = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0
        recalls_large.append(recall_large)

    # Compute AR (Average Recall across IoU 0.5:0.95)
    ar = float(np.mean(recalls)) if recalls else 0.0
    ar_50 = recalls[0] if len(recalls) > 0 else 0.0
    ar_75 = recalls[5] if len(recalls) > 5 else 0.0

    # Compute size-specific AR metrics
    ar_small = float(np.mean(recalls_small)) if recalls_small else 0.0
    ar_medium = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    ar_large = float(np.mean(recalls_large)) if recalls_large else 0.0

    # Create metrics dictionary
    metrics = {
        "AP": -1.0,  # AP not applicable for recall-only evaluation
        "AP50": ar_50,
        "AP75": ar_75,
        "APs": ar_small,
        "APm": ar_medium,
        "APl": ar_large,
        "AR": ar,
        "ARs": ar_small,
        "ARm": ar_medium,
        "ARl": ar_large,
        "AR@0.5": ar_50,
        "AR@0.75": ar_75,
        "model_version": "v3" if is_v3 else "v2"
    }

    print(f"\n{'Metric':<12} {'Value':>9}  {'Description':<50}")
    print("-" * 80)
    print(f"{'AR':<12} {metrics['AR']*100:>9.1f}%  {'Average Recall @ IoU=0.50:0.95':<50}")
    print(f"{'AR@0.5':<12} {metrics['AR@0.5']*100:>9.1f}%  {'Average Recall @ IoU=0.50':<50}")
    print(f"{'AR@0.75':<12} {metrics['AR@0.75']*100:>9.1f}%  {'Average Recall @ IoU=0.75':<50}")
    print(f"{'ARs':<12} {metrics['ARs']*100:>9.1f}%  {'Average Recall (small objects, area < 32²)':<50}")
    print(f"{'ARm':<12} {metrics['ARm']*100:>9.1f}%  {'Average Recall (medium objects, 32² <= area < 96²)':<50}")
    print(f"{'ARl':<12} {metrics['ARl']*100:>9.1f}%  {'Average Recall (large objects, area >= 96²)':<50}")

    # Save results
    print(f"\nSaving results to: {args.result_file}")
    os.makedirs(os.path.dirname(args.result_file) if os.path.dirname(args.result_file) else '.', exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(per_sample_results, f, indent=2)

    # Save metrics
    metrics_file = args.result_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")

    # Save action stats
    action_stats_file = args.result_file.replace('.json', '_action_stats.json')
    action_stats_dict = {k: dict(v) for k, v in action_stats.items()}
    with open(action_stats_file, 'w') as f:
        json.dump(action_stats_dict, f, indent=2)
    print(f"✓ Action stats saved to: {action_stats_file}")

    # Save per-image results if verbose
    if show_verbose:
        per_image_file = args.result_file.replace('.json', '_per_image.json')
        with open(per_image_file, 'w') as f:
            json.dump(per_sample_results, f, indent=2)
        print(f"✓ Per-image results saved to: {per_image_file}")
        
        # Count visualizations
        if viz_dir and os.path.exists(viz_dir):
            num_viz = len([f for f in os.listdir(viz_dir) if f.endswith('.jpg')])
            print(f"✓ Visualizations saved to: {viz_dir}/ ({num_viz} images)")

    # Log final metrics to WandB
    if use_wandb:
        wandb.log({
            'AR': metrics['AR'],
            'AR@0.5': metrics['AR@0.5'],
            'AR@0.75': metrics['AR@0.75'],
            'ARs': metrics['ARs'],
            'ARm': metrics['ARm'],
            'ARl': metrics['ARl'],
            'model_version': 'v3' if is_v3 else 'v2'
        })
        
        # Save artifacts
        wandb.save(args.result_file)
        wandb.save(metrics_file)
        wandb.save(action_stats_file)
        
        # Create action performance table
        action_table_data = []
        for action, stats in sorted(action_stats_dict.items(), key=lambda x: x[1]['total_gt_pairs'], reverse=True)[:20]:
            recall = stats['matched_pairs_05'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
            action_table_data.append([
                action,
                stats['total_samples'],
                stats['total_gt_pairs'],
                stats['matched_pairs_05'],
                f"{recall:.1%}"
            ])
        
        wandb.log({
            "action_performance_table": wandb.Table(
                columns=["Action", "Samples", "GT Pairs", "Matched@0.5", "Recall@0.5"],
                data=action_table_data
            )
        })
        
        wandb.finish()
        print("✓ W&B logging complete")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET/SWIG Grounding Evaluation with Groma Qwen V2/V3")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma Qwen V2 or V3 model checkpoint")
    parser.add_argument("--base-model-name", type=str, default=None,
                        help="Base model to load processor from (optional)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use (GPU ID as integer or 'cuda:X')")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to grounding annotation file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-sample results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="groma-qwen-v3-grounding",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    eval_model(args)

