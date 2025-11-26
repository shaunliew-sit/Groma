"""
HICO-DET Grounding Evaluation Script (Multi-Pair Format)

Evaluates grounding performance on HICO-DET dataset with multi-pair support.

Task: Given "[grounding] Identify the following person and objects in the image: person {action} {object} and the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Format:
- Input: One query per (action, object) combination
- Output: Multi-line format for multiple pairs:
    <p>person</p><roi><r0></roi><p>object</p><roi><r1></roi>
    <p>person</p><roi><r2></roi><p>object</p><roi><r3></roi>

Metrics: Pair-level Precision, Recall, F1 @ IoU thresholds (0.3, 0.5, 0.75)
"""

import os
import json
import re
import torch
import argparse
import torchvision
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.data.datasets.hico_ground import HICOGroundTest, collate_fn
from PIL import ImageDraw, ImageFont

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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

    Args:
        gt_pair: Dict with 'person_box' and 'object_box' keys
        area_small: Threshold for small objects (default: 32² = 1024)
        area_medium: Threshold for medium objects (default: 96² = 9216)

    Returns: 'small', 'medium', or 'large'
    """
    object_area = get_box_area(gt_pair['object_box'])

    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    else:
        return 'large'


def parse_grounding_response(response_text, pred_boxes, img_shape):
    """
    Parse multi-line grounding response to extract person-object pairs.

    Args:
        response_text: Generated text like:
            "<p>person</p><roi><r0></roi><p>object</p><roi><r1></roi>
             <p>person</p><roi><r2></roi><p>object</p><roi><r3></roi>"
        pred_boxes: Tensor of predicted boxes (N, 4) in normalized [cx, cy, w, h] format
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs: [{'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2], ...}, ...]
    """
    pairs = []
    h, w = img_shape

    # Split response into lines
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract region indices from this line
        # Pattern: <p>person</p><roi><r{idx}></roi><p>object</p><roi><r{idx}></roi>
        region_pattern = r'<r(\d+)>'
        matches = re.findall(region_pattern, line)

        if len(matches) >= 2:
            # First region is person, second is object
            person_idx = int(matches[0])
            object_idx = int(matches[1])

            # Validate indices
            if person_idx < len(pred_boxes) and object_idx < len(pred_boxes):
                # Get boxes and convert from cxcywh to xyxy
                person_box = pred_boxes[person_idx]
                object_box = pred_boxes[object_idx]

                person_box_xyxy = torchvision.ops.box_convert(person_box.unsqueeze(0), 'cxcywh', 'xyxy')[0]
                object_box_xyxy = torchvision.ops.box_convert(object_box.unsqueeze(0), 'cxcywh', 'xyxy')[0]

                # Rescale to image dimensions
                person_box_rescaled = [
                    float(person_box_xyxy[0] * w),
                    float(person_box_xyxy[1] * h),
                    float(person_box_xyxy[2] * w),
                    float(person_box_xyxy[3] * h)
                ]
                object_box_rescaled = [
                    float(object_box_xyxy[0] * w),
                    float(object_box_xyxy[1] * h),
                    float(object_box_xyxy[2] * w),
                    float(object_box_xyxy[3] * h)
                ]

                pairs.append({
                    'person_box': person_box_rescaled,
                    'object_box': object_box_rescaled,
                    'person_idx': person_idx,
                    'object_idx': object_idx
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


def visualize_grounding_results(img_path, all_pred_boxes, pred_pairs, gt_pairs, matches,
                                  unmatched_preds, unmatched_gts, output_path, img_shape,
                                  generated_text, action, object_category):
    """
    Visualize grounding results with 3-panel layout.

    Left: All proposed regions from detector
    Middle: Predicted person-object pairs with IoU analysis
    Right: Ground truth person-object pairs
    """
    # Load original image
    img = Image.open(img_path).convert('RGB')

    # Create figure with three subplots
    viz_width = img.width * 3
    viz_height = img.height + 350  # Extra space for legend
    viz_image = Image.new('RGB', (viz_width, viz_height), color='white')

    # Create three panel images
    img_left = img.copy()
    img_middle = img.copy()
    img_right = img.copy()

    draw_left = ImageDraw.Draw(img_left)
    draw_middle = ImageDraw.Draw(img_middle)
    draw_right = ImageDraw.Draw(img_right)

    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        font_tiny = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 8)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()

    # IoU-based colors
    iou_colors = {
        'high': '#00FF00',         # Green: IoU > 0.7 (good match)
        'medium': '#FFA500',       # Orange: 0.5 ≤ IoU ≤ 0.7
        'low': '#FF0000',          # Red: IoU < 0.5
        'unmatched': '#800080'     # Purple: No GT match
    }

    stats = {
        'high_iou_pairs': 0,
        'medium_iou_pairs': 0,
        'low_iou_pairs': 0,
        'unmatched_pred_pairs': len(unmatched_preds),
        'unmatched_gt_pairs': len(unmatched_gts)
    }

    # LEFT PANEL: Draw all proposed regions
    if len(all_pred_boxes) > 0:
        boxes_xyxy = torchvision.ops.box_convert(all_pred_boxes, 'cxcywh', 'xyxy')
        h_img, w_img = img_shape

        # Collect selected indices from predictions
        selected_indices = set()
        for pair in pred_pairs:
            selected_indices.add(pair['person_idx'])
            selected_indices.add(pair['object_idx'])

        for idx, box in enumerate(boxes_xyxy):
            x1 = float(box[0]) * w_img
            y1 = float(box[1]) * h_img
            x2 = float(box[2]) * w_img
            y2 = float(box[3]) * h_img

            # Different color for selected vs unselected
            if idx in selected_indices:
                outline_color = '#FFD700'  # Gold
                width = 3
            else:
                outline_color = '#CCCCCC'  # Gray
                width = 1

            draw_left.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)

            # Draw region index
            label = f"R{idx}"
            try:
                bbox = draw_left.textbbox((x1, y1 - 15), label, font=font_small)
                draw_left.rectangle(bbox, fill=outline_color)
            except:
                pass
            draw_left.text((x1, y1 - 15), label, fill='black', font=font_small)

    # MIDDLE PANEL: Draw predicted pairs with IoU analysis
    for match in matches:
        pred_idx, gt_idx, person_iou, object_iou = match

        pred_pair = pred_pairs[pred_idx]
        avg_iou = (person_iou + object_iou) / 2.0

        # Determine color based on average IoU
        if avg_iou > 0.7:
            box_color = iou_colors['high']
            stats['high_iou_pairs'] += 1
        elif avg_iou >= 0.5:
            box_color = iou_colors['medium']
            stats['medium_iou_pairs'] += 1
        else:
            box_color = iou_colors['low']
            stats['low_iou_pairs'] += 1

        # Draw person box
        px1, py1, px2, py2 = pred_pair['person_box']
        draw_middle.rectangle([px1, py1, px2, py2], outline=box_color, width=3)
        label_p = f"Person(R{pred_pair['person_idx']})\nIoU:{person_iou:.2f}"
        try:
            bbox = draw_middle.textbbox((px1, py1 - 30), label_p, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((px1, py1 - 30), label_p, fill='white', font=font_tiny)

        # Draw object box
        ox1, oy1, ox2, oy2 = pred_pair['object_box']
        draw_middle.rectangle([ox1, oy1, ox2, oy2], outline=box_color, width=3)
        label_o = f"{object_category}(R{pred_pair['object_idx']})\nIoU:{object_iou:.2f}"
        try:
            bbox = draw_middle.textbbox((ox1, oy1 - 30), label_o, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((ox1, oy1 - 30), label_o, fill='white', font=font_tiny)

        # Draw connection line
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
        draw_middle.line([person_center, object_center], fill=box_color, width=2)

    # Draw unmatched predictions
    for pred_idx in unmatched_preds:
        pred_pair = pred_pairs[pred_idx]
        box_color = iou_colors['unmatched']

        # Draw person box
        px1, py1, px2, py2 = pred_pair['person_box']
        draw_middle.rectangle([px1, py1, px2, py2], outline=box_color, width=3)
        label_p = f"Person(R{pred_pair['person_idx']})\n[NO MATCH]"
        try:
            bbox = draw_middle.textbbox((px1, py1 - 30), label_p, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((px1, py1 - 30), label_p, fill='white', font=font_tiny)

        # Draw object box
        ox1, oy1, ox2, oy2 = pred_pair['object_box']
        draw_middle.rectangle([ox1, oy1, ox2, oy2], outline=box_color, width=3)
        label_o = f"{object_category}(R{pred_pair['object_idx']})\n[NO MATCH]"
        try:
            bbox = draw_middle.textbbox((ox1, oy1 - 30), label_o, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((ox1, oy1 - 30), label_o, fill='white', font=font_tiny)

    # RIGHT PANEL: Draw ground truth pairs
    for gt_idx, gt_pair in enumerate(gt_pairs):
        # Check if matched
        is_matched = gt_idx not in unmatched_gts

        # Draw person box
        px1, py1, px2, py2 = gt_pair['person_box']
        if is_matched:
            draw_right.rectangle([px1, py1, px2, py2], outline='green', width=3)
        else:
            # Dashed border for unmatched
            dash_length = 10
            for i in range(int(px1), int(px2), dash_length * 2):
                draw_right.line([(i, py1), (min(i + dash_length, px2), py1)], fill='red', width=3)
                draw_right.line([(i, py2), (min(i + dash_length, px2), py2)], fill='red', width=3)
            for i in range(int(py1), int(py2), dash_length * 2):
                draw_right.line([(px1, i), (px1, min(i + dash_length, py2))], fill='red', width=3)
                draw_right.line([(px2, i), (px2, min(i + dash_length, py2))], fill='red', width=3)

        label_p = f"GT_Person_{gt_idx}"
        color_p = 'green' if is_matched else 'red'
        try:
            bbox = draw_right.textbbox((px1, py1 - 15), label_p, font=font_tiny)
            draw_right.rectangle(bbox, fill=color_p)
        except:
            pass
        draw_right.text((px1, py1 - 15), label_p, fill='white', font=font_tiny)

        # Draw object box
        ox1, oy1, ox2, oy2 = gt_pair['object_box']
        if is_matched:
            draw_right.rectangle([ox1, oy1, ox2, oy2], outline='green', width=3)
        else:
            # Dashed border
            for i in range(int(ox1), int(ox2), dash_length * 2):
                draw_right.line([(i, oy1), (min(i + dash_length, ox2), oy1)], fill='red', width=3)
                draw_right.line([(i, oy2), (min(i + dash_length, ox2), oy2)], fill='red', width=3)
            for i in range(int(oy1), int(oy2), dash_length * 2):
                draw_right.line([(ox1, i), (ox1, min(i + dash_length, oy2))], fill='red', width=3)
                draw_right.line([(ox2, i), (ox2, min(i + dash_length, oy2))], fill='red', width=3)

        label_o = f"GT_{object_category}_{gt_idx}"
        try:
            bbox = draw_right.textbbox((ox1, oy1 - 15), label_o, font=font_tiny)
            draw_right.rectangle(bbox, fill=color_p)
        except:
            pass
        draw_right.text((ox1, oy1 - 15), label_o, fill='white', font=font_tiny)

        # Draw connection line
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
        draw_right.line([person_center, object_center], fill=color_p, width=2)

    # Add titles
    draw_left.text((10, 10), f"All Regions (N={len(all_pred_boxes)})", fill='white', font=font)
    draw_left.rectangle([8, 8, 300, 28], outline='black', width=2)

    draw_middle.text((10, 10), "Predicted Pairs", fill='white', font=font)
    draw_middle.rectangle([8, 8, 200, 28], outline='black', width=2)

    draw_right.text((10, 10), "Ground Truth Pairs", fill='white', font=font)
    draw_right.rectangle([8, 8, 230, 28], outline='black', width=2)

    # Combine panels
    viz_image.paste(img_left, (0, 0))
    viz_image.paste(img_middle, (img.width, 0))
    viz_image.paste(img_right, (img.width * 2, 0))

    # Add legend and statistics
    legend_draw = ImageDraw.Draw(viz_image)
    y_offset = img.height + 10

    # Action-object query
    legend_draw.text((10, y_offset), f"Query: person {action} {object_category}", fill='black', font=font)
    y_offset += 20

    # Generated text
    legend_draw.text((10, y_offset), "Generated Response:", fill='black', font=font)
    y_offset += 18
    truncated_text = generated_text[:180] + "..." if len(generated_text) > 180 else generated_text
    legend_draw.text((10, y_offset), truncated_text, fill='blue', font=font_tiny)
    y_offset += 30

    # Legend
    legend_draw.text((10, y_offset), "Legend:", fill='black', font=font)
    y_offset += 18

    legend_items = [
        (iou_colors['high'], "Green: Good match (Avg IoU > 0.7)"),
        (iou_colors['medium'], "Orange: Partial match (0.5 ≤ Avg IoU ≤ 0.7)"),
        (iou_colors['low'], "Red: Poor match (Avg IoU < 0.5)"),
        (iou_colors['unmatched'], "Purple: No GT match - False positive"),
        ('#FFD700', "Gold outline: Selected regions (left panel)"),
        ('red', "Red dashed: Missed GT pairs (right panel)")
    ]

    for color, text in legend_items:
        legend_draw.rectangle([10, y_offset, 30, y_offset + 12], fill=color, outline='black')
        legend_draw.text((35, y_offset), text, fill='black', font=font_small)
        y_offset += 16

    # Statistics
    y_offset += 8
    legend_draw.text((10, y_offset), "Statistics:", fill='black', font=font)
    y_offset += 18

    total_pred_pairs = len(pred_pairs)
    total_gt_pairs = len(gt_pairs)

    if total_pred_pairs > 0:
        stats_text = [
            f"Total predicted pairs: {total_pred_pairs}",
            f"  Good (Avg IoU > 0.7): {stats['high_iou_pairs']} ({stats['high_iou_pairs']/total_pred_pairs*100:.1f}%)",
            f"  Partial (0.5 ≤ IoU ≤ 0.7): {stats['medium_iou_pairs']} ({stats['medium_iou_pairs']/total_pred_pairs*100:.1f}%)",
            f"  Poor (IoU < 0.5): {stats['low_iou_pairs']} ({stats['low_iou_pairs']/total_pred_pairs*100:.1f}%)",
            f"  No match: {stats['unmatched_pred_pairs']} ({stats['unmatched_pred_pairs']/total_pred_pairs*100:.1f}%)",
            f"Total GT pairs: {total_gt_pairs}",
            f"Missed GT pairs: {stats['unmatched_gt_pairs']}"
        ]

        for text in stats_text:
            legend_draw.text((10, y_offset), text, fill='black', font=font_small)
            y_offset += 14

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz_image.save(output_path)

    return stats


def eval_model(args):
    """Main evaluation function"""

    # Initialize model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f"Loading model from: {model_name}")

    model_short_name = os.path.basename(model_name.rstrip('/'))

    # Get timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres
    model.eval()

    print(f"Model loaded successfully")
    print(f"Model short name: {model_short_name}")
    print(f"Box score threshold: {args.box_score_thres}")

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"hico_ground_{model_short_name}_{timestamp}",
            config={
                "model": model_short_name,
                "dataset": "HICO-DET-Ground",
                "task": "multi_pair_grounding",
                "box_score_threshold": args.box_score_thres,
                "max_images": args.max_images,
                "timestamp": timestamp,
                "annotation_file": args.ann_file,
                "image_prefix": args.img_prefix
            },
            tags=["hico", "grounding", "multi-pair"]
        )
        print(f"✓ Weights & Biases initialized: {wandb.run.url}")

    # Load dataset
    dataset = HICOGroundTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        conv_temp='llava'
    )

    # Limit dataset if requested
    if args.max_images is not None and args.max_images < len(dataset):
        print(f"\n⚠️  Limiting evaluation to first {args.max_images} samples (out of {len(dataset)})")
        dataset.samples = dataset.samples[:args.max_images]

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        sampler=sampler,
        collate_fn=collate_fn
    )

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Each sample = one (action, object) combination")
    print("=" * 80)

    # Evaluation metrics at different IoU thresholds
    # Use extended thresholds for AR computation (0.5 to 0.95 in 0.05 steps)
    iou_thresholds_ar = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # [0.5, 0.55, ..., 0.95]

    # COCO area thresholds for small/medium/large objects
    AREA_SMALL = 32 ** 2    # < 1024 pixels²
    AREA_MEDIUM = 96 ** 2   # < 9216 pixels²

    results_per_threshold = {
        iou_thr: {
            'tp': 0,  # True positives (matched pairs)
            'fp': 0,  # False positives (unmatched predictions)
            'fn': 0,  # False negatives (unmatched GT pairs)
            'tp_small': 0,   # TP for small objects
            'fn_small': 0,   # FN for small objects
            'tp_medium': 0,  # TP for medium objects
            'fn_medium': 0,  # FN for medium objects
            'tp_large': 0,   # TP for large objects
            'fn_large': 0,   # FN for large objects
        }
        for iou_thr in iou_thresholds_ar
    }

    # For backwards compatibility and detailed stats
    iou_thresholds = iou_thresholds_ar

    # Per-sample results
    per_sample_results = []

    # Per-action statistics
    action_stats = defaultdict(lambda: {
        'total_samples': 0,
        'total_gt_pairs': 0,
        'total_pred_pairs': 0,
        'matched_pairs_05': 0
    })

    show_verbose = args.verbose or len(dataset) <= 100

    # Create visualization directory
    viz_output_dir = None
    aggregate_viz_stats = defaultdict(int)
    if show_verbose:
        print("\nVERBOSE MODE: Showing detailed results for each sample\n")
        viz_dir_name = f"visualizations_{model_short_name}_{timestamp}"
        viz_output_dir = os.path.join(os.path.dirname(args.result_file), viz_dir_name)
        os.makedirs(viz_output_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_output_dir}\n")

    print("\nStarting evaluation...")
    for idx, batch in enumerate(tqdm(dataloader, disable=show_verbose)):
        # Extract sample data
        input_ids = batch['input_ids'].unsqueeze(0).cuda()
        image = batch['image'].unsqueeze(0).cuda()
        file_name = batch['file_name']
        action = batch['action']
        object_category = batch['object_category']
        gt_boxes_flat = batch['gt_boxes']  # List of boxes [x1,y1,x2,y2]
        num_pairs = batch['num_pairs']
        img_shape = (batch['height'], batch['width'])
        action_object_id = batch['action_object_id']

        # Build GT pairs (person, object alternating)
        gt_pairs = []
        for i in range(0, len(gt_boxes_flat), 2):
            if i + 1 < len(gt_boxes_flat):
                gt_pairs.append({
                    'person_box': gt_boxes_flat[i],
                    'object_box': gt_boxes_flat[i + 1]
                })

        if show_verbose:
            print(f"\n[Sample {idx+1}/{len(dataset)}] {file_name}")
            print(f"  Action: {action}, Object: {object_category}")
            print(f"  GT pairs: {len(gt_pairs)}")

        # Run model inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=200,  # Allow for multiple pairs
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

        # Extract predictions
        output_ids = outputs.sequences
        pred_boxes_all = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()

        # Decode generated text
        input_token_len = input_ids.shape[1]
        generated_text = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=False)

        if show_verbose:
            print(f"  Response: {generated_text[:150]}...")

        # Parse predicted pairs
        pred_pairs = parse_grounding_response(generated_text, pred_boxes_all, img_shape)

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
            'action_object_id': action_object_id,
            'num_gt_pairs': len(gt_pairs),
            'num_pred_pairs': len(pred_pairs),
            'generated_text': generated_text[:200],
            'matches_per_threshold': {}
        }

        for iou_thr in iou_thresholds:
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

                if show_verbose:
                    print(f"  Matched @ IoU=0.5: {len(matches)}/{len(gt_pairs)}")

        per_sample_results.append(sample_result)

        # Log per-sample metrics to WandB
        if use_wandb:
            recall_05 = len([m for m, _, _, _ in match_pairs_greedy(pred_pairs, gt_pairs, 0.5)[0]]) / len(gt_pairs) if len(gt_pairs) > 0 else 0.0
            wandb.log({
                'sample_idx': idx,
                'recall@0.5': recall_05,
                'num_pred_pairs': len(pred_pairs),
                'num_gt_pairs': len(gt_pairs),
                'num_matched@0.5': sample_result['matches_per_threshold'][0.5]['matched'],
            })

        # Create visualization
        if show_verbose and viz_output_dir is not None:
            img_path = batch['img_path']
            # Include action and object in filename to avoid overwriting when same image has multiple HOI triplets
            safe_action = action.replace(' ', '_')
            safe_object = object_category.replace(' ', '_')
            viz_path = os.path.join(viz_output_dir, f"{os.path.splitext(file_name)[0]}_{safe_action}_{safe_object}_ground.jpg")

            # Get matches at IoU=0.5 for visualization
            matches_05, unmatched_preds_05, unmatched_gts_05 = match_pairs_greedy(
                pred_pairs, gt_pairs, iou_threshold=0.5
            )

            try:
                viz_stats = visualize_grounding_results(
                    img_path=img_path,
                    all_pred_boxes=pred_boxes_all,
                    pred_pairs=pred_pairs,
                    gt_pairs=gt_pairs,
                    matches=matches_05,
                    unmatched_preds=unmatched_preds_05,
                    unmatched_gts=unmatched_gts_05,
                    output_path=viz_path,
                    img_shape=img_shape,
                    generated_text=generated_text,
                    action=action,
                    object_category=object_category
                )

                # Aggregate stats
                for key in viz_stats:
                    aggregate_viz_stats[key] += viz_stats[key]

                print(f"  💾 Visualization saved to: {viz_path}")

                # Upload visualization to WandB
                if use_wandb:
                    try:
                        wandb.log({
                            f"visualizations/{file_name}": wandb.Image(
                                viz_path,
                                caption=f"Sample {idx+1}: {action} {object_category} | {len(matches_05)}/{len(gt_pairs)} matched"
                            )
                        })
                    except Exception as e:
                        print(f"  ⚠️  WandB visualization upload failed: {e}")
            except Exception as e:
                print(f"  ⚠️  Visualization failed: {e}")

    # Compute Average Recall (AR) metrics
    print("\n" + "=" * 80)
    print("HICO-DET Grounding Evaluation Results (Multi-Pair Format)")
    print("=" * 80)

    # Compute recalls at all IoU thresholds
    recalls = []
    for iou_thr in iou_thresholds_ar:
        tp = results_per_threshold[iou_thr]['tp']
        fn = results_per_threshold[iou_thr]['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    # Compute AR (Average Recall across IoU 0.5:0.95)
    ar = float(np.mean(recalls)) if recalls else 0.0

    # AR at specific IoU thresholds
    ar_50 = recalls[0] if len(recalls) > 0 else 0.0  # IoU=0.5 (index 0)
    ar_75 = recalls[5] if len(recalls) > 5 else 0.0  # IoU=0.75 (index 5)

    # Compute size-based AR (ARs, ARm, ARl)
    # Uses object box area for categorization (COCO standard)
    # Small: area < 32^2 = 1024, Medium: 1024 <= area < 96^2 = 9216, Large: area >= 9216
    recalls_small = []
    recalls_medium = []
    recalls_large = []

    for iou_thr in iou_thresholds_ar:
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

    # Compute size-specific AR metrics (average across IoU thresholds)
    ar_small = float(np.mean(recalls_small)) if recalls_small else 0.0
    ar_medium = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    ar_large = float(np.mean(recalls_large)) if recalls_large else 0.0

    # Create metrics dictionary in desired format
    metrics = {
        "AP": -1.0,  # AP not applicable for recall-only evaluation
        "AP50": ar_50,  # Using AR@0.5 as proxy
        "AP75": ar_75,  # Using AR@0.75 as proxy
        "APs": ar_small,
        "APm": ar_medium,
        "APl": ar_large,
        "AR": ar,
        "ARs": ar_small,
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
    print(f"{'ARs':<12} {metrics['ARs']*100:>9.1f}%  {'Small pairs (area < 32²)':<50}")
    print(f"{'ARm':<12} {metrics['ARm']*100:>9.1f}%  {'Medium pairs (32² < area < 96²)':<50}")
    print(f"{'ARl':<12} {metrics['ARl']*100:>9.1f}%  {'Large pairs (area > 96²)':<50}")
    print("-" * 80)
    print(f"\nNote: AP metrics set to -1.0 (not applicable for recall-only evaluation)")
    print(f"      Size-based AR uses object box area for categorization (COCO standard)")

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        wandb.log({
            'total_samples': len(dataset),
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

    if use_wandb:
        # Log summary files as artifacts
        wandb.save(metrics_file)
        wandb.save(args.result_file)
        wandb.save(action_stats_file)
        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET Grounding Evaluation (Multi-Pair)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma model checkpoint")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to HICO grounding annotation file (e.g., hico_ground_test.json)")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--box-score-thres", type=float, default=0.15,
                        help="Box confidence threshold")
    parser.add_argument("--batch-size-per-gpu", type=int, default=1,
                        help="Batch size (must be 1 for evaluation)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-sample results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="groma-hico-ground",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size_per_gpu != 1:
        print("Warning: batch_size_per_gpu must be 1 for evaluation, setting to 1")
        args.batch_size_per_gpu = 1

    eval_model(args)
