"""
HICO-DET Grounding Evaluation Script for InternVL3 (Multi-Pair Format)

Evaluates InternVL3 grounding performance on HICO-DET dataset with multi-pair support.

Key Differences from Qwen3VL Evaluation:
- Uses InternVL3 model instead of Qwen3VL
- Uses InternVL3's native image preprocessing (dynamic tiling)
- Same prompt format and JSON output parsing
- Same coordinate format: [0, 1000] normalized → pixel conversion
- Same metrics: AR (Average Recall) at multiple IoU thresholds

Task: Given "Detect all person-{object} pairs where the person is {action} the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Output Format (from InternVL3):
[
  {"pair_id": 1, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]},
  {"pair_id": 2, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}
]

Metrics: Pair-level Precision, Recall, F1 @ IoU thresholds (0.5 to 0.95)
"""

import os
import json
import re
import math
import torch
import argparse
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import InterpolationMode
import numpy as np

from transformers import AutoModel, AutoTokenizer

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# InternVL3 Image Preprocessing Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build image transformation pipeline for InternVL3."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
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
    """Dynamically preprocess image into multiple tiles for InternVL3."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

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


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

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


def parse_internvl3_json_response(response_text, img_shape):
    """
    Parse InternVL3 JSON response to extract person-object pairs.

    InternVL3 uses the same [0, 1000] normalized coordinate format as Qwen3VL.
    Handles multiple formats:
    - {"person": [x1,y1,x2,y2], "bench": [x1,y1,x2,y2]}
    - {"person_bbox": [x1,y1,x2,y2], "object_bbox": [x1,y1,x2,y2]}

    Args:
        response_text: Generated JSON text
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs with pixel coordinates
    """
    pairs = []
    h, w = img_shape

    # Clean markdown code fences
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = '\n'.join(lines)

    # Try JSON parsing
    try:
        data = json.loads(cleaned_text)

        # Handle list format
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Try multiple key formats
                    person_bbox_norm = item.get('person_bbox') or item.get('person') or item.get('person_box')
                    # For object, also try all keys except 'person*' and 'pair_id'
                    object_bbox_norm = item.get('object_bbox') or item.get('object') or item.get('object_box')
                    if not object_bbox_norm:
                        # Try to find any key that's not person-related
                        for key, value in item.items():
                            if key not in ['person', 'person_bbox', 'person_box', 'pair_id'] and isinstance(value, list):
                                object_bbox_norm = value
                                break

                    if person_bbox_norm and object_bbox_norm and \
                       len(person_bbox_norm) == 4 and len(object_bbox_norm) == 4:
                        # Convert from [0, 1000] to pixel coordinates
                        person_box = [
                            (person_bbox_norm[0] / 1000.0) * w,
                            (person_bbox_norm[1] / 1000.0) * h,
                            (person_bbox_norm[2] / 1000.0) * w,
                            (person_bbox_norm[3] / 1000.0) * h
                        ]
                        object_box = [
                            (object_bbox_norm[0] / 1000.0) * w,
                            (object_bbox_norm[1] / 1000.0) * h,
                            (object_bbox_norm[2] / 1000.0) * w,
                            (object_bbox_norm[3] / 1000.0) * h
                        ]

                        pairs.append({
                            'person_box': person_box,
                            'object_box': object_box,
                            'pair_id': item.get('pair_id', len(pairs) + 1)
                        })

        # Handle dict format (single pair or pairs list)
        elif isinstance(data, dict):
            if 'pairs' in data:
                # Nested format: {"pairs": [...]}
                return parse_internvl3_json_response(json.dumps(data['pairs']), img_shape)
            elif 'person_bbox' in data or 'person' in data:
                # Single pair format
                person_bbox_norm = data.get('person_bbox') or data.get('person') or data.get('person_box')
                object_bbox_norm = data.get('object_bbox') or data.get('object') or data.get('object_box')
                if not object_bbox_norm:
                    # Try to find any key that's not person-related
                    for key, value in data.items():
                        if key not in ['person', 'person_bbox', 'person_box', 'pair_id'] and isinstance(value, list):
                            object_bbox_norm = value
                            break

                if person_bbox_norm and object_bbox_norm and \
                   len(person_bbox_norm) == 4 and len(object_bbox_norm) == 4:
                    person_box = [
                        (person_bbox_norm[0] / 1000.0) * w,
                        (person_bbox_norm[1] / 1000.0) * h,
                        (person_bbox_norm[2] / 1000.0) * w,
                        (person_bbox_norm[3] / 1000.0) * h
                    ]
                    object_box = [
                        (object_bbox_norm[0] / 1000.0) * w,
                        (object_bbox_norm[1] / 1000.0) * h,
                        (object_bbox_norm[2] / 1000.0) * w,
                        (object_bbox_norm[3] / 1000.0) * h
                    ]

                    pairs.append({
                        'person_box': person_box,
                        'object_box': object_box,
                        'pair_id': 1
                    })

    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        # Fallback: regex extraction
        pair_pattern = r'"person_bbox"?\s*:\s*\[([^\]]+)\].*?"object_bbox"?\s*:\s*\[([^\]]+)\]'
        matches = re.findall(pair_pattern, cleaned_text, re.DOTALL)

        for person_str, object_str in matches:
            try:
                person_coords = [float(x.strip()) for x in person_str.split(',')]
                object_coords = [float(x.strip()) for x in object_str.split(',')]

                if len(person_coords) == 4 and len(object_coords) == 4:
                    person_box = [
                        (person_coords[0] / 1000.0) * w,
                        (person_coords[1] / 1000.0) * h,
                        (person_coords[2] / 1000.0) * w,
                        (person_coords[3] / 1000.0) * h
                    ]
                    object_box = [
                        (object_coords[0] / 1000.0) * w,
                        (object_coords[1] / 1000.0) * h,
                        (object_coords[2] / 1000.0) * w,
                        (object_coords[3] / 1000.0) * h
                    ]

                    pairs.append({
                        'person_box': person_box,
                        'object_box': object_box,
                        'pair_id': len(pairs) + 1
                    })
            except (ValueError, IndexError):
                continue

    return pairs


def match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5):
    """
    Match predicted pairs to ground truth pairs using greedy matching.

    A pair matches if BOTH person AND object boxes have IoU > threshold.
    """
    matches = []
    unmatched_preds = list(range(len(pred_pairs)))
    unmatched_gts = list(range(len(gt_pairs)))

    # Calculate all pair IoUs
    pair_ious = []
    for pred_idx, pred_pair in enumerate(pred_pairs):
        for gt_idx, gt_pair in enumerate(gt_pairs):
            person_iou = calculate_iou(pred_pair['person_box'], gt_pair['person_box'])
            object_iou = calculate_iou(pred_pair['object_box'], gt_pair['object_box'])

            # Both must exceed threshold
            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                avg_iou = (person_iou + object_iou) / 2.0
                pair_ious.append((avg_iou, pred_idx, gt_idx, person_iou, object_iou))

    # Sort by average IoU (descending)
    pair_ious.sort(reverse=True, key=lambda x: x[0])

    # Greedy matching
    for avg_iou, pred_idx, gt_idx, person_iou, object_iou in pair_ious:
        if pred_idx in unmatched_preds and gt_idx in unmatched_gts:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'person_iou': person_iou,
                'object_iou': object_iou,
                'avg_iou': avg_iou
            })
            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(gt_idx)

    return matches, unmatched_preds, unmatched_gts


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
        image_path: Path to image
        action: Action being performed
        object_category: Object category
        max_num: Maximum number of image tiles
        device: Device to use

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


def visualize_internvl3_grounding(image, pred_pairs, gt_pairs, output_path,
                                  action, object_category, iou_threshold=0.5):
    """Visualize InternVL3 grounding results with 3-panel comparison."""
    fig_width = image.width * 3 + 60
    fig_height = image.height + 120

    fig_img = Image.new('RGB', (fig_width, fig_height), color='white')
    draw = ImageDraw.Draw(fig_img)

    # Try to load font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Match pairs
    matches, unmatched_preds, unmatched_gts = match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold)

    # Draw titles
    draw.text((10, 10), f"Predictions ({len(pred_pairs)})", fill='black', font=title_font)
    draw.text((image.width + 30, 10), f"Ground Truth ({len(gt_pairs)})", fill='black', font=title_font)
    draw.text((image.width * 2 + 50, 10), f"Overlay (Matches: {len(matches)})", fill='black', font=title_font)

    # Panel 1: Predictions
    pred_img = image.copy()
    pred_draw = ImageDraw.Draw(pred_img)
    for idx, pair in enumerate(pred_pairs):
        color = '#00FF00' if idx not in unmatched_preds else '#FF0000'
        pred_draw.rectangle(pair['person_box'], outline=color, width=3)
        pred_draw.rectangle(pair['object_box'], outline=color, width=3)
        # Connect with line
        p_center = ((pair['person_box'][0] + pair['person_box'][2]) / 2,
                    (pair['person_box'][1] + pair['person_box'][3]) / 2)
        o_center = ((pair['object_box'][0] + pair['object_box'][2]) / 2,
                    (pair['object_box'][1] + pair['object_box'][3]) / 2)
        pred_draw.line([p_center, o_center], fill=color, width=2)
    fig_img.paste(pred_img, (10, 50))

    # Panel 2: Ground Truth
    gt_img = image.copy()
    gt_draw = ImageDraw.Draw(gt_img)
    for idx, pair in enumerate(gt_pairs):
        color = '#0000FF'
        gt_draw.rectangle(pair['person_box'], outline=color, width=3)
        gt_draw.rectangle(pair['object_box'], outline=color, width=3)
        p_center = ((pair['person_box'][0] + pair['person_box'][2]) / 2,
                    (pair['person_box'][1] + pair['person_box'][3]) / 2)
        o_center = ((pair['object_box'][0] + pair['object_box'][2]) / 2,
                    (pair['object_box'][1] + pair['object_box'][3]) / 2)
        gt_draw.line([p_center, o_center], fill=color, width=2)
    fig_img.paste(gt_img, (image.width + 30, 50))

    # Panel 3: Overlay
    overlay_img = image.copy()
    overlay_draw = ImageDraw.Draw(overlay_img)
    # Draw matches in green
    for match in matches:
        pred_pair = pred_pairs[match['pred_idx']]
        overlay_draw.rectangle(pred_pair['person_box'], outline='#00FF00', width=3)
        overlay_draw.rectangle(pred_pair['object_box'], outline='#00FF00', width=3)
    # Draw unmatched predictions in red
    for idx in unmatched_preds:
        pred_pair = pred_pairs[idx]
        overlay_draw.rectangle(pred_pair['person_box'], outline='#FF0000', width=2)
        overlay_draw.rectangle(pred_pair['object_box'], outline='#FF0000', width=2)
    # Draw unmatched GTs in blue
    for idx in unmatched_gts:
        gt_pair = gt_pairs[idx]
        overlay_draw.rectangle(gt_pair['person_box'], outline='#0000FF', width=2)
        overlay_draw.rectangle(gt_pair['object_box'], outline='#0000FF', width=2)
    fig_img.paste(overlay_img, (image.width * 2 + 50, 50))

    # Add stats at bottom
    stats_y = image.height + 60
    stats_text = f"Action: {action} | Object: {object_category} | IoU Threshold: {iou_threshold:.2f}"
    draw.text((10, stats_y), stats_text, fill='black', font=text_font)
    stats_text2 = f"Matches: {len(matches)} | FP: {len(unmatched_preds)} | FN: {len(unmatched_gts)}"
    draw.text((10, stats_y + 25), stats_text2, fill='black', font=text_font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig_img.save(output_path)


def eval_model(args):
    """Main evaluation function."""
    print("="*80)
    print("HICO-DET Grounding Evaluation (InternVL3)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Annotation file: {args.ann_file}")
    print(f"Image prefix: {args.img_prefix}")
    print(f"Result file: {args.result_file}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print("="*80)
    print()

    # Initialize WandB
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project or "hico-grounding-internvl3",
            name=args.wandb_run_name or f"internvl3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": args.model_name,
                "dataset": "HICO-DET",
                "task": "grounding",
                "max_images": args.max_images
            }
        )

    # Load model
    print("Loading InternVL3 model...")
    print(f"Target device: {args.device}")

    # Handle device mapping
    # When CUDA_VISIBLE_DEVICES is set, we need to use the local device index
    # e.g., if CUDA_VISIBLE_DEVICES=5, then cuda:0 refers to physical GPU 5
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

    # Load dataset
    print(f"Loading annotation file: {args.ann_file}")
    with open(args.ann_file, 'r') as f:
        dataset_samples = json.load(f)

    print(f"✓ Loaded {len(dataset_samples)} samples\n")

    if args.max_images:
        dataset_samples = dataset_samples[:args.max_images]
        print(f"Limited to {len(dataset_samples)} samples for testing\n")

    # Create visualization directory
    viz_dir = None
    if args.verbose:
        viz_dir = args.result_file.replace('.json', '_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}\n")

    # Evaluation loop
    results = []
    results_per_threshold = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

    # Size-based results (COCO standard: small < 32^2, medium 32^2-96^2, large > 96^2)
    results_per_size = {
        'small': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds},
        'medium': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds},
        'large': {iou_thr: {'tp': 0, 'fp': 0, 'fn': 0} for iou_thr in iou_thresholds}
    }

    print("Starting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Evaluating")):
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
        gt_box_pairs = []
        for conv in sample['conversation']:
            if conv['from'] == 'gpt' and conv.get('box_inds'):
                box_inds = conv['box_inds']
                # Pairs are: [person1, object1, person2, object2, ...]
                for i in range(0, len(box_inds), 2):
                    if i + 1 < len(box_inds):
                        person_idx = box_inds[i]
                        object_idx = box_inds[i + 1]
                        if person_idx < len(boxes) and object_idx < len(boxes):
                            gt_box_pairs.append({
                                'person_box': boxes[person_idx],
                                'object_box': boxes[object_idx]
                            })

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

        # Match at all IoU thresholds
        matches_per_threshold = {}
        for iou_thr in iou_thresholds:
            matches, unmatched_preds, unmatched_gts = match_pairs_greedy(
                pred_pairs, gt_box_pairs, iou_thr
            )
            results_per_threshold[iou_thr]['tp'] += len(matches)
            results_per_threshold[iou_thr]['fp'] += len(unmatched_preds)
            results_per_threshold[iou_thr]['fn'] += len(unmatched_gts)

            # Track size-based metrics using average size of person and object boxes
            for match in matches:
                gt_idx = match['gt_idx']
                gt_pair = gt_box_pairs[gt_idx]
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
                if pred_idx < len(pred_pairs):
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

            # Track unmatched ground truths by size
            for gt_idx in unmatched_gts:
                if gt_idx < len(gt_box_pairs):
                    gt_pair = gt_box_pairs[gt_idx]
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

        # Store result
        results.append({
            'file_name': file_name,
            'action': action,
            'object': object_category,
            'action_object_id': action_object_id,
            'num_gt_pairs': len(gt_box_pairs),
            'num_pred_pairs': len(pred_pairs),
            'prompt': prompt_text,  # Include prompt for debugging
            'generated_text': output_text[:200],
            'matches_per_threshold': matches_per_threshold
        })

        # Visualize
        if viz_dir and idx < 20:  # Limit visualizations
            viz_path = os.path.join(viz_dir, f"{idx:04d}_{action}_{object_category}.jpg")
            visualize_internvl3_grounding(
                image, pred_pairs, gt_box_pairs, viz_path,
                action, object_category, iou_threshold=0.5
            )

        # Log to WandB
        if use_wandb and idx % 10 == 0:
            wandb.log({
                'samples_processed': idx + 1,
                'current_sample': file_name
            })

    # Calculate metrics
    print("\n" + "="*80)
    print("Computing Metrics")
    print("="*80)

    recalls = []
    for iou_thr in iou_thresholds:
        tp = results_per_threshold[iou_thr]['tp']
        fn = results_per_threshold[iou_thr]['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
        print(f"Recall @ IoU={iou_thr:.2f}: {recall:.4f}")

    ar = float(np.mean(recalls)) if recalls else 0.0
    ar_50 = recalls[0] if len(recalls) > 0 else 0.0
    ar_75 = recalls[5] if len(recalls) > 5 else 0.0

    # Compute size-specific AR metrics
    print("\n" + "="*80)
    print("Computing Size-Specific Metrics")
    print("="*80)

    ar_small_list = []
    ar_medium_list = []
    ar_large_list = []

    for iou_thr in iou_thresholds:
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

    print(f"AR (small):  {ar_small:.4f}")
    print(f"AR (medium): {ar_medium:.4f}")
    print(f"AR (large):  {ar_large:.4f}")

    # Create metrics dictionary (matches SWIG format)
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

    print(f"\n{'='*80}")
    print("Final Metrics")
    print("="*80)
    print(f"{'Metric':<12} {'Value':>9}  {'Description':<50}")
    print("-" * 80)
    print(f"{'AR':<12} {metrics['AR']*100:>9.1f}%  {'Average Recall @ IoU=0.50:0.95':<50}")
    print(f"{'AR@0.5':<12} {metrics['AR@0.5']*100:>9.1f}%  {'Average Recall @ IoU=0.50':<50}")
    print(f"{'AR@0.75':<12} {metrics['AR@0.75']*100:>9.1f}%  {'Average Recall @ IoU=0.75':<50}")
    print(f"{'ARs':<12} {metrics['ARs']*100:>9.1f}%  {'AR for small objects (area < 32^2)':<50}")
    print(f"{'ARm':<12} {metrics['ARm']*100:>9.1f}%  {'AR for medium objects (32^2 <= area <= 96^2)':<50}")
    print(f"{'ARl':<12} {metrics['ARl']*100:>9.1f}%  {'AR for large objects (area > 96^2)':<50}")
    print("="*80)

    # Save results
    output_data = {
        'model': args.model_name,
        'dataset': 'HICO-DET',
        'task': 'grounding',
        'metrics': metrics,
        'per_sample_results': results,
        'timestamp': datetime.now().isoformat()
    }

    with open(args.result_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Results saved to: {args.result_file}")

    # Save metrics
    metrics_file = args.result_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        wandb.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser(description='HICO-DET Grounding Evaluation for InternVL3')
    parser.add_argument('--model-name', type=str, default='OpenGVLab/InternVL3-8B',
                        help='InternVL3 model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0, cuda:1, auto)')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to HICO grounding annotation file')
    parser.add_argument('--img-prefix', type=str, required=True,
                        help='Path to HICO images directory')
    parser.add_argument('--result-file', type=str, required=True,
                        help='Output result file path')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limit to first N images')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed per-sample results')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='WandB run name')

    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()
