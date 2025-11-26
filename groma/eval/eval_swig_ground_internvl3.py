"""
SWIG-HOI Grounding Evaluation Script for InternVL3 (Multi-Pair Format)

Evaluates InternVL3 grounding performance on SWIG-HOI dataset with multi-pair support.

Key Differences from Qwen3VL Evaluation:
- Uses InternVL3 model instead of Qwen3VL
- Uses InternVL3 dynamic image preprocessing (multi-tile approach)
- Parses InternVL3 output format (<ref></ref><box></box> or JSON)
- Direct bbox prediction in [0, 1000] format (same as Qwen3VL)
- Same metrics: AR (Average Recall) at multiple IoU thresholds
- Supports person-person interactions

Task: Given "Detect all person-{object} pairs where the person is {action} the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Output Format (from InternVL3):
[
  {"pair_id": 1, "person": [x1, y1, x2, y2], "{object}": [x1, y1, x2, y2]},
  {"pair_id": 2, "person": [x1, y1, x2, y2], "{object}": [x1, y1, x2, y2]}
]

Metrics: Pair-level Precision, Recall, F1 @ IoU thresholds (0.5 to 0.95)
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# InternVL3 Image Preprocessing Functions
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build image transform for InternVL3."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio from target_ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image into multiple tiles.
    This is InternVL3's approach to handling various aspect ratios.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split into grid
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL3."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================================
# Utility Functions
# ============================================================================

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


def parse_internvl3_json_response(response_text, img_shape):
    """
    Parse InternVL3 JSON response to extract person-object pairs.

    InternVL3 uses [0, 1000] normalized coordinates (same as Qwen3VL).
    Handles multiple formats:
    - {"person": [x1,y1,x2,y2], "bench": [x1,y1,x2,y2]}
    - {"person_bbox": [x1,y1,x2,y2], "object_bbox": [x1,y1,x2,y2]}

    Args:
        response_text: Generated JSON text like:
            ```json
            [
              {"person": [x1, y1, x2, y2], "bench": [x1, y1, x2, y2]},
              {"person": [x1, y1, x2, y2], "bench": [x1, y1, x2, y2]}
            ]
            ```
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs: [{'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2]}, ...]
        Boxes are in pixel coordinates
    """
    pairs = []
    h, w = img_shape

    # Clean markdown code fences if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = '\n'.join(lines)

    # Try to parse JSON
    try:
        parsed = json.loads(cleaned_text.strip())
        if isinstance(parsed, list):
            detections = parsed
        elif isinstance(parsed, dict):
            detections = [parsed]
        else:
            return pairs
    except json.JSONDecodeError:
        # Fallback: try regex extraction
        array_pattern = r'\[[\s\S]+\]'
        array_matches = re.findall(array_pattern, cleaned_text)
        if array_matches:
            longest_match = max(array_matches, key=len)
            try:
                parsed = json.loads(longest_match)
                if isinstance(parsed, list):
                    detections = parsed
                else:
                    return pairs
            except:
                return pairs
        else:
            return pairs

    # Convert InternVL3 format [0, 1000] to pixel coordinates
    for det in detections:
        # Find person and object keys (object key may vary)
        person_key = None
        object_key = None

        for key in det.keys():
            if key.lower() in ['person', 'person_bbox']:
                person_key = key
            elif key.lower() not in ['pair_id', 'person', 'person_bbox']:
                # Assume any other key is the object category
                object_key = key

        if person_key and object_key:
            person_bbox_internvl = det[person_key]
            object_bbox_internvl = det[object_key]

            if len(person_bbox_internvl) == 4 and len(object_bbox_internvl) == 4:
                # Convert from [0, 1000] to pixel coordinates
                person_box = [
                    (person_bbox_internvl[0] / 1000.0) * w,
                    (person_bbox_internvl[1] / 1000.0) * h,
                    (person_bbox_internvl[2] / 1000.0) * w,
                    (person_bbox_internvl[3] / 1000.0) * h
                ]

                object_box = [
                    (object_bbox_internvl[0] / 1000.0) * w,
                    (object_bbox_internvl[1] / 1000.0) * h,
                    (object_bbox_internvl[2] / 1000.0) * w,
                    (object_bbox_internvl[3] / 1000.0) * h
                ]

                pairs.append({
                    'person_box': person_box,
                    'object_box': object_box,
                    'pair_id': det.get('pair_id', len(pairs))
                })

    return pairs


def get_box_area(box):
    """Calculate area of bounding box [x1, y1, x2, y2]."""
    return (box[2] - box[0]) * (box[3] - box[1])


def get_size_category(box):
    """
    Categorize box by size following COCO standard.
    - small: area < 32^2 (1024)
    - medium: 32^2 <= area <= 96^2 (1024-9216)
    - large: area > 96^2 (9216)
    """
    area = get_box_area(box)
    if area < 1024:
        return 'small'
    elif area <= 9216:
        return 'medium'
    else:
        return 'large'


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


def visualize_internvl3_grounding(image_path, pred_pairs, gt_pairs, matches,
                                  action, object_category, iou_threshold=0.5):
    """
    Create visualization comparing predicted pairs vs ground truth pairs.

    Args:
        image_path: Path to image file
        pred_pairs: List of predicted pairs with 'person_box' and 'object_box'
        gt_pairs: List of ground truth pairs
        matches: List of (pred_idx, gt_idx, person_iou, object_iou) from matching
        action: Action verb
        object_category: Object category
        iou_threshold: IoU threshold used for matching

    Returns:
        PIL Image with 3-panel visualization (predictions | ground truth | overlay)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Create 3 copies for 3-panel visualization
    pred_img = image.copy()
    gt_img = image.copy()
    overlay_img = image.copy()

    # Create drawing contexts
    pred_draw = ImageDraw.Draw(pred_img)
    gt_draw = ImageDraw.Draw(gt_img)
    overlay_draw = ImageDraw.Draw(overlay_img)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Color scheme
    color_person_pred = (255, 0, 0)      # Red for predicted person
    color_object_pred = (0, 0, 255)      # Blue for predicted object
    color_person_gt = (0, 255, 0)        # Green for GT person
    color_object_gt = (255, 255, 0)      # Yellow for GT object
    color_matched = (0, 255, 0)          # Green for matched pairs
    color_unmatched = (255, 0, 0)        # Red for unmatched pairs

    # Draw predicted pairs (left panel)
    matched_pred_indices = {m[0] for m in matches}
    for idx, pred_pair in enumerate(pred_pairs):
        is_matched = idx in matched_pred_indices

        # Person box
        person_box = pred_pair['person_box']
        pred_draw.rectangle(person_box, outline=color_person_pred, width=3)
        pred_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}",
                      fill=color_person_pred, font=small_font)

        # Object box
        object_box = pred_pair['object_box']
        pred_draw.rectangle(object_box, outline=color_object_pred, width=3)
        pred_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}",
                      fill=color_object_pred, font=small_font)

        # Draw line connecting pair
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        line_color = color_matched if is_matched else color_unmatched
        pred_draw.line([person_center, object_center], fill=line_color, width=2)

    # Draw ground truth pairs (middle panel)
    matched_gt_indices = {m[1] for m in matches}
    for idx, gt_pair in enumerate(gt_pairs):
        is_matched = idx in matched_gt_indices

        # Person box
        person_box = gt_pair['person_box']
        gt_draw.rectangle(person_box, outline=color_person_gt, width=3)
        gt_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}",
                    fill=color_person_gt, font=small_font)

        # Object box
        object_box = gt_pair['object_box']
        gt_draw.rectangle(object_box, outline=color_object_gt, width=3)
        gt_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}",
                    fill=color_object_gt, font=small_font)

        # Draw line connecting pair
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        line_color = color_matched if is_matched else color_unmatched
        gt_draw.line([person_center, object_center], fill=line_color, width=2)

    # Draw overlay (right panel) - show both predictions and GT with matching
    for match in matches:
        pred_idx, gt_idx, person_iou, object_iou = match
        pred_pair = pred_pairs[pred_idx]
        gt_pair = gt_pairs[gt_idx]

        # Draw GT in green
        overlay_draw.rectangle(gt_pair['person_box'], outline=(0, 255, 0), width=2)
        overlay_draw.rectangle(gt_pair['object_box'], outline=(0, 255, 0), width=2)

        # Draw prediction in blue (dashed effect with thinner line)
        overlay_draw.rectangle(pred_pair['person_box'], outline=(0, 100, 255), width=2)
        overlay_draw.rectangle(pred_pair['object_box'], outline=(0, 100, 255), width=2)

        # Show IoU scores
        avg_iou = (person_iou + object_iou) / 2.0
        overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                         f"IoU:{avg_iou:.2f}", fill=(255, 255, 255), font=small_font)

    # Draw unmatched predictions in red
    for idx in range(len(pred_pairs)):
        if idx not in matched_pred_indices:
            pred_pair = pred_pairs[idx]
            overlay_draw.rectangle(pred_pair['person_box'], outline=(255, 0, 0), width=2)
            overlay_draw.rectangle(pred_pair['object_box'], outline=(255, 0, 0), width=2)
            overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                            "FP", fill=(255, 0, 0), font=small_font)

    # Draw unmatched GT in orange
    for idx in range(len(gt_pairs)):
        if idx not in matched_gt_indices:
            gt_pair = gt_pairs[idx]
            overlay_draw.rectangle(gt_pair['person_box'], outline=(255, 165, 0), width=2)
            overlay_draw.rectangle(gt_pair['object_box'], outline=(255, 165, 0), width=2)
            overlay_draw.text((gt_pair['person_box'][0], gt_pair['person_box'][1] - 18),
                            "FN", fill=(255, 165, 0), font=small_font)

    # Create final 3-panel image
    panel_width = img_width
    panel_height = img_height
    total_width = panel_width * 3
    header_height = 60
    total_height = panel_height + header_height

    final_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    final_draw = ImageDraw.Draw(final_img)

    # Add headers
    title_y = 10
    final_draw.text((panel_width // 2 - 100, title_y), "Predictions", fill=(0, 0, 0), font=font)
    final_draw.text((panel_width + panel_width // 2 - 80, title_y), "Ground Truth", fill=(0, 0, 0), font=font)
    final_draw.text((2 * panel_width + panel_width // 2 - 60, title_y), "Overlay", fill=(0, 0, 0), font=font)

    # Add statistics
    stats_y = 35
    num_matched = len(matches)
    num_fp = len(pred_pairs) - num_matched
    num_fn = len(gt_pairs) - num_matched
    recall = num_matched / len(gt_pairs) if len(gt_pairs) > 0 else 0.0

    stats_text = f"Action: {action} | Object: {object_category} | IoU≥{iou_threshold}"
    final_draw.text((10, stats_y), stats_text, fill=(0, 0, 0), font=small_font)

    metrics_text = f"Pred:{len(pred_pairs)} | GT:{len(gt_pairs)} | Matched:{num_matched} | FP:{num_fp} | FN:{num_fn} | Recall:{recall:.1%}"
    final_draw.text((panel_width + 10, stats_y), metrics_text, fill=(0, 0, 0), font=small_font)

    # Paste images
    final_img.paste(pred_img, (0, header_height))
    final_img.paste(gt_img, (panel_width, header_height))
    final_img.paste(overlay_img, (2 * panel_width, header_height))

    return final_img


def build_internvl3_grounding_prompt(action, object_category):
    """
    Build prompt for InternVL3 grounding task.

    Follows the exact format from test_internvl_grounding.sh with <ref> tags.
    Format: <ref>person {action} {object}</ref> and <ref>{object}</ref>
    Example: <ref>person sitting on bench</ref> and <ref>bench</ref>
    """
    # Match test_internvl_grounding.sh format exactly for single pair,
    # but clarify it should return array if multiple pairs exist
    prompt_text = (
        f"<image>\n"
        f"Identify the following person and objects in the image: "
        f"<ref>person {action} {object_category}</ref> and <ref>{object_category}</ref>. "
        f'Return JSON format: {{"person": [x1, y1, x2, y2], "{object_category}": [x1, y1, x2, y2]}}. '
        f"If multiple pairs exist, return array: "
        f'[{{"person": [x1, y1, x2, y2], "{object_category}": [x1, y1, x2, y2]}}, '
        f'{{"person": [x1, y1, x2, y2], "{object_category}": [x1, y1, x2, y2]}}].'
    )
    return prompt_text


def run_internvl3_inference(model, tokenizer, image_path, action, object_category,
                            max_num=12, device=None):
    """
    Run InternVL3 inference for grounding task.

    Args:
        model: InternVL3 model
        tokenizer: InternVL3 tokenizer
        image_path: Path to image file
        action: Action verb
        object_category: Object category name
        max_num: Maximum number of image tiles
        device: Device for computation (unused, kept for API compatibility)

    Returns:
        (output_text, image): Generated text and PIL Image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    pixel_values = load_image_internvl(image_path, max_num=max_num)

    # Move to device - use .cuda() like test script
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # Build prompt - includes <image> tag in the prompt text
    prompt_text = build_internvl3_grounding_prompt(action, object_category)

    # Generate response
    generation_config = dict(
        max_new_tokens=512,  # Support multiple pairs
        do_sample=False      # Deterministic
    )

    response = model.chat(tokenizer, pixel_values, prompt_text, generation_config)

    return response, image


def eval_model(args):
    """Main evaluation function"""

    print("=" * 80)
    print("SWIG-HOI Grounding Evaluation (InternVL3)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Device:      {args.device}")
    print(f"Annotation:  {args.ann_file}")
    print(f"Images:      {args.img_prefix}")
    print(f"Output:      {args.result_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    print("=" * 80)
    print()

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load InternVL3 model
    print(f"Loading InternVL3 model: {args.model_name}")
    print(f"Target device: {args.device}")

    # Handle device mapping
    if args.device == "auto":
        device_map = "auto"
    elif args.device.startswith("cuda"):
        # Extract device index
        if ":" in args.device:
            device_idx = args.device.split(":")[1]
        else:
            device_idx = "0"

        # Check if CUDA_VISIBLE_DEVICES is set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if cuda_visible is not None:
            # When CUDA_VISIBLE_DEVICES is set, use cuda:0 (the first visible device)
            device_map = {"": "cuda:0"}
            print(f"  Note: CUDA_VISIBLE_DEVICES={cuda_visible}, using cuda:0 (physical GPU {cuda_visible})")
        else:
            # No CUDA_VISIBLE_DEVICES, use specified device directly
            device_map = {"": f"cuda:{device_idx}"}
    else:
        device_map = {"": args.device}

    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False
    )

    # Get actual device
    device = next(model.parameters()).device
    print(f"✓ Model loaded successfully on device: {device}\n")

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")

        # Check if user is logged in
        try:
            # Try to login (will use cached credentials if available)
            wandb.login()

            # Initialize run
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"swig_ground_internvl3_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "dataset": "SWIG-HOI-Ground",
                    "task": "multi_pair_grounding",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["swig", "grounding", "internvl3", "multi-pair", "person-person"]
            )
            print(f"✓ Weights & Biases initialized successfully!")
            print(f"  Run URL: {wandb.run.url}")
            print(f"  Project: {wandb.run.project}")
            print(f"  Run name: {wandb.run.name}\n")
        except Exception as e:
            print(f"⚠️  Warning: WandB initialization failed: {e}")
            print(f"  To use WandB, please run: wandb login")
            print(f"  Continuing evaluation without WandB logging...\n")
            use_wandb = False

    # Load annotation file
    print(f"Loading annotations from: {args.ann_file}")
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

    results_per_threshold = {
        iou_thr: {
            'tp': 0,  # True positives (matched pairs)
            'fp': 0,  # False positives (unmatched predictions)
            'fn': 0,  # False negatives (unmatched GT pairs)
        }
        for iou_thr in iou_thresholds_ar
    }

    # Size-based results (COCO standard: small < 32^2, medium 32^2-96^2, large > 96^2)
    results_per_size = {
        'small': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds_ar},
        'medium': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds_ar},
        'large': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds_ar}
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
        print(f"✓ Visualization directory: {viz_dir}\n")

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        file_name = sample['file_name']
        action = sample['action']
        object_category = sample['object_category']
        action_object_id = sample.get('action_object_id', f"{action}_{object_category}")

        img_path = os.path.join(args.img_prefix, file_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        # Get image shape
        with Image.open(img_path) as img:
            img_shape = (img.height, img.width)

        # Build GT pairs from conversation
        # box_inds are in the 'gpt' message and contain [person1, object1, person2, object2, ...]
        boxes = sample['boxes']
        gt_pairs = []
        for conv in sample['conversation']:
            if conv['from'] == 'gpt' and conv.get('box_inds'):
                box_inds = conv['box_inds']
                # Pairs are: [person1, object1, person2, object2, ...]
                for i in range(0, len(box_inds), 2):
                    if i + 1 < len(box_inds):
                        person_idx = box_inds[i]
                        object_idx = box_inds[i + 1]
                        if person_idx < len(boxes) and object_idx < len(boxes):
                            gt_pairs.append({
                                'person_box': boxes[person_idx],
                                'object_box': boxes[object_idx]
                            })

        if show_verbose:
            print(f"\n[Sample {idx+1}/{len(dataset_samples)}] {file_name}")
            print(f"  Action: {action}, Object: {object_category}")
            print(f"  GT pairs: {len(gt_pairs)}")

        # Build and log prompt (for debugging)
        prompt_text = build_internvl3_grounding_prompt(action, object_category)

        # Run inference
        try:
            output_text, image = run_internvl3_inference(
                model, tokenizer, img_path, action, object_category,
                max_num=12, device=device
            )

            # Parse predictions
            pred_pairs = parse_internvl3_json_response(output_text, img_shape)

        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            output_text = ""
            pred_pairs = []
            prompt_text = ""

        if show_verbose:
            print(f"  Response: {output_text[:150]}...")
            print(f"  Predicted pairs: {len(pred_pairs)}")

        # Update action stats
        action_stats[action]['total_samples'] += 1
        action_stats[action]['total_gt_pairs'] += len(gt_pairs)
        action_stats[action]['total_pred_pairs'] += len(pred_pairs)

        # Match at all IoU thresholds
        matches_per_threshold = {}
        for iou_thr in iou_thresholds_ar:
            matches, unmatched_preds, unmatched_gts = match_pairs_greedy(
                pred_pairs, gt_pairs, iou_threshold=iou_thr
            )
            results_per_threshold[iou_thr]['tp'] += len(matches)
            results_per_threshold[iou_thr]['fp'] += len(unmatched_preds)
            results_per_threshold[iou_thr]['fn'] += len(unmatched_gts)

            # Track size-based metrics using average size of person and object boxes
            for match_idx, gt_idx, _, _ in matches:
                gt_pair = gt_pairs[gt_idx]
                # Use average area of person and object boxes to determine size
                person_area = get_box_area(gt_pair['person_box'])
                object_area = get_box_area(gt_pair['object_box'])
                avg_area = (person_area + object_area) / 2.0

                # Determine size category based on average area
                if avg_area < 1024:
                    size_cat = 'small'
                elif avg_area <= 9216:
                    size_cat = 'medium'
                else:
                    size_cat = 'large'

                results_per_size[size_cat][iou_thr]['tp'] += 1

            # Track unmatched predictions by size
            for pred_idx in unmatched_preds:
                pred_pair = pred_pairs[pred_idx]
                person_area = get_box_area(pred_pair['person_box'])
                object_area = get_box_area(pred_pair['object_box'])
                avg_area = (person_area + object_area) / 2.0

                if avg_area < 1024:
                    size_cat = 'small'
                elif avg_area <= 9216:
                    size_cat = 'medium'
                else:
                    size_cat = 'large'

                results_per_size[size_cat][iou_thr]['fp'] += 1

            # Track unmatched GT by size
            for gt_idx in unmatched_gts:
                gt_pair = gt_pairs[gt_idx]
                person_area = get_box_area(gt_pair['person_box'])
                object_area = get_box_area(gt_pair['object_box'])
                avg_area = (person_area + object_area) / 2.0

                if avg_area < 1024:
                    size_cat = 'small'
                elif avg_area <= 9216:
                    size_cat = 'medium'
                else:
                    size_cat = 'large'

                results_per_size[size_cat][iou_thr]['fn'] += 1

            matches_per_threshold[f'iou_{iou_thr:.2f}'] = {
                'matched': len(matches),
                'unmatched_preds': len(unmatched_preds),
                'unmatched_gts': len(unmatched_gts)
            }

            if iou_thr == 0.5:
                action_stats[action]['matched_pairs_05'] += len(matches)

        if show_verbose:
            matched_05 = matches_per_threshold['iou_0.50']['matched']
            print(f"  Matched @ IoU=0.5: {matched_05}/{len(gt_pairs)}")

        # Store result
        per_sample_results.append({
            'file_name': file_name,
            'action': action,
            'object': object_category,
            'action_object_id': action_object_id,
            'num_gt_pairs': len(gt_pairs),
            'num_pred_pairs': len(pred_pairs),
            'prompt': prompt_text,  # Log the prompt for debugging
            'generated_text': output_text[:200],
            'matches_per_threshold': matches_per_threshold
        })

        # Visualize - save all visualizations if viz_dir is set
        if viz_dir:
            viz_path = os.path.join(viz_dir, f"{idx:04d}_{action}_{object_category}.jpg")
            try:
                # Get matches at IoU=0.5 for visualization
                matches_05_list, _, _ = match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5)
                viz_img = visualize_internvl3_grounding(
                    img_path, pred_pairs, gt_pairs, matches_05_list,
                    action, object_category, iou_threshold=0.5
                )
                viz_img.save(viz_path, quality=90)
            except Exception as e:
                if show_verbose:
                    print(f"  Warning: Visualization failed: {e}")

        # Log to WandB
        if use_wandb and idx % 10 == 0:
            wandb.log({
                'samples_processed': idx + 1,
                'current_sample': file_name
            })

    # Compute Average Recall (AR) metrics
    print("\n" + "=" * 80)
    print("SWIG-HOI Grounding Evaluation Results (InternVL3)")
    print("=" * 80)

    # Compute recalls at all IoU thresholds (overall)
    recalls = []
    for iou_thr in iou_thresholds_ar:
        tp = results_per_threshold[iou_thr]['tp']
        fn = results_per_threshold[iou_thr]['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    # Compute AR (Average Recall across IoU 0.5:0.95)
    ar = float(np.mean(recalls)) if recalls else 0.0
    ar_50 = recalls[0] if len(recalls) > 0 else 0.0
    ar_75 = recalls[5] if len(recalls) > 5 else 0.0

    # Compute size-specific AR metrics
    ar_small_list = []
    ar_medium_list = []
    ar_large_list = []

    for iou_thr in iou_thresholds_ar:
        # Small
        tp_s = results_per_size['small'][iou_thr]['tp']
        fn_s = results_per_size['small'][iou_thr]['fn']
        recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        ar_small_list.append(recall_s)

        # Medium
        tp_m = results_per_size['medium'][iou_thr]['tp']
        fn_m = results_per_size['medium'][iou_thr]['fn']
        recall_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
        ar_medium_list.append(recall_m)

        # Large
        tp_l = results_per_size['large'][iou_thr]['tp']
        fn_l = results_per_size['large'][iou_thr]['fn']
        recall_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
        ar_large_list.append(recall_l)

    ar_small = float(np.mean(ar_small_list)) if ar_small_list else 0.0
    ar_medium = float(np.mean(ar_medium_list)) if ar_medium_list else 0.0
    ar_large = float(np.mean(ar_large_list)) if ar_large_list else 0.0

    # Create metrics dictionary
    metrics = {
        "AP": -1.0,  # AP not applicable for recall-only evaluation
        "AP50": ar_50,
        "AP75": ar_75,
        "APs": ar_small,  # Size-specific AP (approximated from AR)
        "APm": ar_medium,
        "APl": ar_large,
        "AR": ar,
        "ARs": ar_small,  # Size-specific AR
        "ARm": ar_medium,
        "ARl": ar_large,
        "AR@0.5": ar_50,
        "AR@0.75": ar_75
    }

    # Print detailed results
    print("\n" + "=" * 80)
    print("Average Recall (AR) Metrics")
    print("=" * 80)
    print(f"{'Metric':<12} {'Value':>10}  {'Description':<50}")
    print("-" * 80)
    print(f"{'AR':<12} {metrics['AR']*100:>9.1f}%  {'Average Recall @ IoU=0.50:0.95':<50}")
    print(f"{'AR@0.5':<12} {metrics['AR@0.5']*100:>9.1f}%  {'Average Recall @ IoU=0.50':<50}")
    print(f"{'AR@0.75':<12} {metrics['AR@0.75']*100:>9.1f}%  {'Average Recall @ IoU=0.75':<50}")
    print(f"{'ARs':<12} {metrics['ARs']*100:>9.1f}%  {'AR for small objects (area < 32^2)':<50}")
    print(f"{'ARm':<12} {metrics['ARm']*100:>9.1f}%  {'AR for medium objects (32^2 <= area <= 96^2)':<50}")
    print(f"{'ARl':<12} {metrics['ARl']*100:>9.1f}%  {'AR for large objects (area > 96^2)':<50}")
    print("-" * 80)

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        wandb.log({
            'total_samples': len(dataset_samples),
            'total_gt_pairs': sum(s['num_gt_pairs'] for s in per_sample_results),
            'total_pred_pairs': sum(s['num_pred_pairs'] for s in per_sample_results),
        })

    # Per-action statistics
    print("\n" + "=" * 80)
    print("Per-Action Statistics (Top 20 by sample count)")
    print("=" * 80)
    print(f"{'Action':<20} {'Samples':>8} {'GT Pairs':>9} {'Pred':>9} {'Matched':>9} {'Recall@0.5':>11}")
    print("-" * 80)

    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['total_samples'], reverse=True)
    for action, stats in sorted_actions[:20]:
        recall_05 = stats['matched_pairs_05'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
        print(f"{action:<20} {stats['total_samples']:>8} {stats['total_gt_pairs']:>9} "
              f"{stats['total_pred_pairs']:>9} {stats['matched_pairs_05']:>9} {recall_05 * 100:>10.1f}%")

    print("=" * 80)

    # Save results
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    # Save metrics
    metrics_file = args.result_file.replace('.json', '_metrics.json')
    print(f"\nSaving metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save per-sample results
    print(f"Saving per-sample results to: {args.result_file}")
    with open(args.result_file, 'w') as f:
        json.dump(per_sample_results, f, indent=2)

    # Save per-action stats
    action_stats_file = args.result_file.replace('.json', '_action_stats.json')
    print(f"Saving per-action stats to: {action_stats_file}")
    action_stats_dict = {action: stats for action, stats in action_stats.items()}
    with open(action_stats_file, 'w') as f:
        json.dump(action_stats_dict, f, indent=2)

    # Save and log visualizations summary
    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith('.jpg')])
        print(f"\nVisualizations: {viz_dir}/")
        print(f"  Total images saved: {viz_count} (should match {len(dataset_samples)} samples)")
        if viz_count != len(dataset_samples):
            print(f"  ⚠️  WARNING: Expected {len(dataset_samples)} visualizations but got {viz_count}")

    if use_wandb:
        wandb.save(metrics_file)
        wandb.save(args.result_file)
        wandb.save(action_stats_file)

        # Create summary table for top/bottom performing actions
        action_table_data = []
        sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['total_samples'], reverse=True)
        for action, stats in sorted_actions[:20]:
            recall_05 = stats['matched_pairs_05'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
            action_table_data.append([
                action,
                stats['total_samples'],
                stats['total_gt_pairs'],
                stats['total_pred_pairs'],
                stats['matched_pairs_05'],
                f"{recall_05:.1%}"
            ])

        wandb.log({
            "action_performance_table": wandb.Table(
                columns=["Action", "Samples", "GT Pairs", "Pred Pairs", "Matched@0.5", "Recall@0.5"],
                data=action_table_data
            )
        })

        wandb.finish()
        print("✓ WandB logging complete")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWIG-HOI Grounding Evaluation with InternVL3")
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternVL3-8B",
                        help="InternVL3 model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to SWIG grounding annotation file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to SWIG images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-sample results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="swig-grounding-internvl3",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    eval_model(args)
