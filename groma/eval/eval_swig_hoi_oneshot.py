"""
SWIG-HOI One-Shot HOI Grounding Evaluation Script

Evaluates one-shot HOI grounding performance on SWIG-HOI dataset.
Task: Given a single prompt, predict ALL bounding boxes for entities in HOI pairs.

Prompt: "[grounding] Describe what each person is doing with objects individually. Focus on actions only."
Expected Response: "The <p>person</p><roi><r9></roi> is <p>sitting on</p> the <p>bench</p><roi><r5></roi>."

Metrics: COCO-style Average Recall (AR, AR@0.5, AR@0.75)
"""

import os
import json
import re
import torch
import argparse
import torchvision
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.data.datasets.swig_hoi_oneshot import SWIGHOIOneShotTest, collate_fn

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# Entity name synonyms for handling mismatches
ENTITY_SYNONYMS = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle',
    'scooter': 'motorcycle',
    'table': 'dining table',
    'desk': 'dining table',
    'ball': 'sports ball',
    'baseball': 'sports ball',
    'soccer ball': 'sports ball',
    'football': 'sports ball',
    'sofa': 'couch',
    'tv monitor': 'tv',
    'television': 'tv',
    'cellphone': 'cell phone',
    'mobile phone': 'cell phone',
    'phone': 'cell phone',
}

# Common HOI actions to filter from entity parsing
HOI_ACTIONS = {
    'sitting on', 'riding', 'holding', 'wearing', 'carrying',
    'eating', 'drinking', 'looking at', 'playing', 'kicking',
    'hitting', 'throwing', 'catching', 'cutting', 'opening',
    'closing', 'pushing', 'pulling', 'washing', 'cleaning',
    'no interaction', 'using', 'touching', 'standing on',
    'lying on', 'walking', 'running', 'jumping', 'sitting',
    'loading', 'straddling', 'racing'
}


def normalize_entity_name(name):
    """Normalize entity name to match COCO categories"""
    name = name.lower().strip()
    return ENTITY_SYNONYMS.get(name, name)


def is_action_word(word):
    """Check if word is an action (not an entity)"""
    return word.lower().strip() in HOI_ACTIONS


def parse_hoi_response(response_text):
    """
    Parse one-shot HOI response to extract all entities and region indices.

    Pattern: <p>entity_name</p><roi><rN></roi>

    Example:
    "The <p>person</p><roi><r9></roi> is <p>sitting on</p> the <p>bench</p><roi><r5></roi>.
     Another <p>person</p><roi><r2></roi> is <p>riding</p> a <p>motorcycle</p><roi><r1></roi>."

    Returns:
        entities: List of entity names ["person", "bench", "person", "motorcycle"]
        region_indices: List of region indices [9, 5, 2, 1]
    """
    # Pattern: <p>entity_name</p> followed by <roi><rN></roi>
    # Use non-greedy matching to handle multiple instances
    pattern = r'<p>\s*([^<]+?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'

    matches = re.findall(pattern, response_text)

    entities = []
    region_indices = []

    for entity_name, region_idx in matches:
        entity_name = entity_name.strip()

        # Skip action words (sitting on, riding, etc.)
        if is_action_word(entity_name):
            continue

        entities.append(entity_name)
        region_indices.append(int(region_idx))

    return entities, region_indices


def deduplicate_boxes(region_indices, entities):
    """
    Remove duplicate region indices while preserving order.
    Handles Hungarian matching artifacts where model generates [r9, r9, r5].
    """
    seen = set()
    unique_indices = []
    unique_entities = []

    for idx, entity in zip(region_indices, entities):
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
            unique_entities.append(entity)

    return unique_indices, unique_entities


def compute_iou_matrix(pred_boxes, gt_boxes):
    """
    Compute IoU matrix between all predicted and ground truth boxes.

    Args:
        pred_boxes: List of predicted boxes in [x, y, w, h] format
        gt_boxes: List of ground truth boxes in [x, y, w, h] format

    Returns:
        iou_matrix: Numpy array of shape [N_pred, N_gt]
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)

    return iou_matrix


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_gt(pred_boxes, pred_entities, gt_boxes, gt_entities):
    """
    Match predicted boxes to ground truth using greedy IoU matching.
    Handles edge cases: name mismatches, more GT than preds, etc.

    Args:
        pred_boxes: List of predicted boxes [x, y, w, h]
        pred_entities: List of predicted entity names
        gt_boxes: List of ground truth boxes [x, y, w, h]
        gt_entities: List of ground truth entity names (COCO categories)

    Returns:
        matches: List of (pred_idx, gt_idx, iou, name_match) tuples
        unmatched_preds: List of pred indices with no GT match
        unmatched_gts: List of GT indices with no pred match
    """
    # Normalize entity names
    pred_entities_norm = [normalize_entity_name(e) for e in pred_entities]
    gt_entities_norm = [normalize_entity_name(e) for e in gt_entities]

    # Build IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # Greedy matching (highest priority first)
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Build candidates with priority
    candidates = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            iou = iou_matrix[i, j]
            if iou > 0.5:  # IoU threshold
                name_match = (pred_entities_norm[i] == gt_entities_norm[j])
                # Prioritize matches with correct name AND high IoU
                priority = (2 if name_match else 1) * iou
                candidates.append((priority, i, j, iou, name_match))

    # Sort by priority (name matches with high IoU come first)
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Assign matches greedily
    for _, pred_idx, gt_idx, iou, name_match in candidates:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx, iou, name_match))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Find unmatched
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def rescale_box(boxes, img_shape):
    """
    Rescale normalized boxes to absolute pixel coordinates.

    Args:
        boxes: Tensor of shape (N, 4) in normalized [cx, cy, w, h] format
        img_shape: Tuple of (height, width)

    Returns:
        Rescaled boxes in [x, y, w, h] format (absolute pixels)
    """
    h, w = img_shape
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * w  # cx
    boxes[:, 1] = boxes[:, 1] * h  # cy
    boxes[:, 2] = boxes[:, 2] * w  # w
    boxes[:, 3] = boxes[:, 3] * h  # h
    return boxes


def visualize_oneshot_results(img_path, all_pred_boxes, pred_boxes, pred_entities, pred_region_indices,
                                gt_boxes, gt_entities, matches, unmatched_preds, unmatched_gts,
                                output_path, img_shape, generated_text):
    """
    Visualize one-shot HOI grounding results with 3-panel layout.

    Left: All proposed regions from detector
    Middle: Model predictions with IoU analysis
    Right: Ground truth boxes
    """
    # Load original image
    img = Image.open(img_path).convert('RGB')

    # Create figure with three subplots
    viz_width = img.width * 3
    viz_height = img.height + 300  # Extra space for legend
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
        'high': '#00FF00',        # Green: IoU > 0.7 AND name match
        'high_mismatch': '#FFFF00',  # Yellow: IoU > 0.7 BUT name mismatch
        'medium': '#FFA500',      # Orange: 0.3 < IoU ≤ 0.7
        'low': '#FF0000',         # Red: IoU ≤ 0.3
        'unmatched': '#800080'    # Purple: No GT match
    }

    stats = {
        'high_iou_name_match': 0,
        'high_iou_name_mismatch': 0,
        'medium_iou': 0,
        'low_iou': 0,
        'unmatched_pred': len(unmatched_preds),
        'unmatched_gt': len(unmatched_gts)
    }

    # LEFT PANEL: Draw all proposed regions
    if len(all_pred_boxes) > 0:
        boxes_xyxy = torchvision.ops.box_convert(all_pred_boxes, 'cxcywh', 'xyxy')
        h_img, w_img = img_shape

        # Highlight selected regions
        selected_indices = set(pred_region_indices)

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

    # MIDDLE PANEL: Draw predictions with IoU analysis
    for match in matches:
        pred_idx, gt_idx, iou, name_match = match

        pred_box = pred_boxes[pred_idx]
        pred_entity = pred_entities[pred_idx]
        region_idx = pred_region_indices[pred_idx]

        gt_entity = gt_entities[gt_idx]

        # Determine color based on IoU and name match
        if iou > 0.7:
            if name_match:
                box_color = iou_colors['high']
                stats['high_iou_name_match'] += 1
            else:
                box_color = iou_colors['high_mismatch']
                stats['high_iou_name_mismatch'] += 1
        elif iou > 0.3:
            box_color = iou_colors['medium']
            stats['medium_iou'] += 1
        else:
            box_color = iou_colors['low']
            stats['low_iou'] += 1

        # Draw box
        x, y, w_box, h_box = pred_box
        x2, y2 = x + w_box, y + h_box

        draw_middle.rectangle([x, y, x2, y2], outline=box_color, width=3)

        # Label
        label = f"{pred_entity}(R{region_idx})\nIoU:{iou:.2f}"
        if not name_match:
            label += f"\n[GT:{gt_entity}]"

        try:
            bbox = draw_middle.textbbox((x, y - 40), label, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((x, y - 40), label, fill='white', font=font_tiny)

    # Draw unmatched predictions
    for pred_idx in unmatched_preds:
        pred_box = pred_boxes[pred_idx]
        pred_entity = pred_entities[pred_idx]
        region_idx = pred_region_indices[pred_idx]

        box_color = iou_colors['unmatched']

        x, y, w_box, h_box = pred_box
        x2, y2 = x + w_box, y + h_box

        draw_middle.rectangle([x, y, x2, y2], outline=box_color, width=3)

        label = f"{pred_entity}(R{region_idx})\n[NO MATCH]"
        try:
            bbox = draw_middle.textbbox((x, y - 30), label, font=font_tiny)
            draw_middle.rectangle(bbox, fill=box_color)
        except:
            pass
        draw_middle.text((x, y - 30), label, fill='white', font=font_tiny)

    # RIGHT PANEL: Draw ground truth
    for gt_idx, (gt_box, gt_entity) in enumerate(zip(gt_boxes, gt_entities)):
        x, y, w_box, h_box = gt_box
        x2, y2 = x + w_box, y + h_box

        # Dashed border for unmatched GT
        if gt_idx in unmatched_gts:
            # Draw dashed
            dash_length = 10
            for i in range(int(x), int(x2), dash_length * 2):
                draw_right.line([(i, y), (min(i + dash_length, x2), y)], fill='red', width=3)
                draw_right.line([(i, y2), (min(i + dash_length, x2), y2)], fill='red', width=3)
            for i in range(int(y), int(y2), dash_length * 2):
                draw_right.line([(x, i), (x, min(i + dash_length, y2))], fill='red', width=3)
                draw_right.line([(x2, i), (x2, min(i + dash_length, y2))], fill='red', width=3)
        else:
            draw_right.rectangle([x, y, x2, y2], outline='green', width=3)

        label = f"GT_{gt_entity}_{gt_idx}"
        try:
            bbox = draw_right.textbbox((x, y - 15), label, font=font_tiny)
            draw_right.rectangle(bbox, fill='green' if gt_idx not in unmatched_gts else 'red')
        except:
            pass
        draw_right.text((x, y - 15), label, fill='white', font=font_tiny)

    # Add titles
    draw_left.text((10, 10), f"All Regions (N={len(all_pred_boxes)})", fill='white', font=font)
    draw_left.rectangle([8, 8, 300, 28], outline='black', width=2)

    draw_middle.text((10, 10), "Predictions with IoU", fill='white', font=font)
    draw_middle.rectangle([8, 8, 250, 28], outline='black', width=2)

    draw_right.text((10, 10), "Ground Truth", fill='white', font=font)
    draw_right.rectangle([8, 8, 180, 28], outline='black', width=2)

    # Combine panels
    viz_image.paste(img_left, (0, 0))
    viz_image.paste(img_middle, (img.width, 0))
    viz_image.paste(img_right, (img.width * 2, 0))

    # Add legend and statistics
    legend_draw = ImageDraw.Draw(viz_image)
    y_offset = img.height + 10

    # Generated text
    legend_draw.text((10, y_offset), "Generated Response:", fill='black', font=font)
    y_offset += 20
    truncated_text = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
    legend_draw.text((10, y_offset), truncated_text, fill='blue', font=font_tiny)
    y_offset += 30

    # Legend
    legend_draw.text((10, y_offset), "Legend:", fill='black', font=font)
    y_offset += 20

    legend_items = [
        (iou_colors['high'], "Green: Good match (IoU>0.7, name match)"),
        (iou_colors['high_mismatch'], "Yellow: Good IoU (>0.7) but name mismatch"),
        (iou_colors['medium'], "Orange: Partial match (0.3<IoU≤0.7) - Region issue"),
        (iou_colors['low'], "Red: Poor match (IoU≤0.3) - Wrong object"),
        (iou_colors['unmatched'], "Purple: No GT match - False positive"),
        ('#FFD700', "Gold outline: Selected regions (left panel)")
    ]

    for color, text in legend_items:
        legend_draw.rectangle([10, y_offset, 30, y_offset + 15], fill=color, outline='black')
        legend_draw.text((35, y_offset), text, fill='black', font=font_small)
        y_offset += 18

    # Statistics
    y_offset += 10
    legend_draw.text((10, y_offset), "Statistics:", fill='black', font=font)
    y_offset += 20

    total_preds = len(pred_boxes)
    if total_preds > 0:
        stats_text = [
            f"Total predictions: {total_preds}",
            f"  Good (IoU>0.7, name match): {stats['high_iou_name_match']} ({stats['high_iou_name_match']/total_preds*100:.1f}%)",
            f"  Good IoU, name mismatch: {stats['high_iou_name_mismatch']} ({stats['high_iou_name_mismatch']/total_preds*100:.1f}%)",
            f"  Partial (0.3<IoU≤0.7): {stats['medium_iou']} ({stats['medium_iou']/total_preds*100:.1f}%)",
            f"  Wrong (IoU≤0.3): {stats['low_iou']} ({stats['low_iou']/total_preds*100:.1f}%)",
            f"  No match: {stats['unmatched_pred']} ({stats['unmatched_pred']/total_preds*100:.1f}%)",
            f"Missed GT boxes: {stats['unmatched_gt']}"
        ]

        for text in stats_text:
            legend_draw.text((10, y_offset), text, fill='black', font=font_small)
            y_offset += 16

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz_image.save(output_path)

    return stats


def convert_to_coco_format(predictions, dataset):
    """Convert predictions to COCO format for evaluation"""
    results = []
    cat_name_to_id = dataset.name_to_cat_id

    for pred in predictions:
        img_id = pred['img_id']
        entities = pred['entities']
        boxes = pred['boxes']
        scores = pred['scores']

        for entity, box, score in zip(entities, boxes, scores):
            # Normalize entity name
            entity_norm = normalize_entity_name(entity)

            if entity_norm not in cat_name_to_id:
                continue

            cat_id = cat_name_to_id[entity_norm]

            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": box,  # [x, y, w, h]
                "score": score
            })

    return results


def create_coco_ground_truth(dataset):
    """Create COCO-format ground truth annotations"""
    images = []
    annotations = []
    categories = []

    # Build categories
    cat_name_to_id = dataset.name_to_cat_id
    for cat_name, cat_id in cat_name_to_id.items():
        categories.append({
            "id": cat_id,
            "name": cat_name
        })

    # Build images and annotations
    ann_id = 1
    for item in dataset.processed_data:
        img_id = item['img_id']

        # Add image info
        images.append({
            "id": img_id,
            "file_name": item['file_name'],
            "width": item['width'],
            "height": item['height']
        })

        # Add annotations
        for box, cat_name in zip(item['all_boxes'], item['all_categories']):
            if cat_name not in cat_name_to_id:
                continue

            cat_id = cat_name_to_id[cat_name]
            x, y, w, h = box

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    return {
        "info": {
            "description": "SWIG-HOI One-Shot HOI Grounding Ground Truth",
            "version": "1.0",
            "year": 2024
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def eval_model(args):
    """Main evaluation function"""

    # Initialize model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f"Loading model from: {model_name}")

    model_short_name = os.path.basename(model_name.rstrip('/'))

    # Get timestamp
    import re
    from datetime import datetime
    timestamp_match = re.search(r'_(\d{8}_\d{6})', args.result_file)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres
    model.eval()

    print(f"Model loaded successfully")
    print(f"Box score threshold: {args.box_score_thres}")

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"swig_hoi_oneshot_{model_short_name}_{timestamp}",
            config={
                "model": model_short_name,
                "dataset": "SWIG-HOI",
                "task": "one_shot_hoi_grounding",
                "box_score_threshold": args.box_score_thres,
                "max_images": args.max_images,
                "timestamp": timestamp,
                "annotation_file": args.ann_file,
                "image_prefix": args.img_prefix
            },
            tags=["swig", "hoi", "grounding", "one-shot"]
        )
        print(f"✓ Weights & Biases initialized: {wandb.run.url}")
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠️  WandB requested but not installed. Install with: pip install wandb")

    # Load dataset
    dataset = SWIGHOIOneShotTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='llava'
    )

    # Limit dataset if requested
    if args.max_images is not None and args.max_images < len(dataset):
        print(f"\n⚠️  Limiting evaluation to first {args.max_images} images (out of {len(dataset)})")
        dataset.processed_data = dataset.processed_data[:args.max_images]

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=4,
        sampler=sampler,
        collate_fn=collate_fn
    )

    print(f"\nStarting inference on {len(dataset)} images...")
    print("=" * 70)

    # Run inference
    predictions = []
    per_image_results = []
    aggregate_stats = defaultdict(int)

    show_verbose = args.verbose or len(dataset) <= 100

    # Create visualization directory
    viz_output_dir = None
    if show_verbose:
        viz_dir_name = f"visualizations_{model_short_name}_{timestamp}"
        viz_output_dir = os.path.join(os.path.dirname(args.result_file), viz_dir_name)
        os.makedirs(viz_output_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_output_dir}\n")

    for idx, (gt_boxes, gt_categories, img_id, img_shape, file_name, base_item) in enumerate(tqdm(dataloader, disable=show_verbose)):
        # Load and preprocess image
        img_path = os.path.join(args.img_prefix, file_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((448, 448), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32)

        # Normalize
        mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
        std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).contiguous()
        image = img_tensor.unsqueeze(0).cuda()

        # Build prompt
        input_ids, full_prompt = dataset.build_prompt()
        input_ids = input_ids.cuda()

        if show_verbose:
            print(f"\n[Image {idx+1}/{len(dataset)}] {file_name}")
            print(f"  Image ID: {img_id}")
            print(f"  GT boxes: {len(gt_boxes)}")

        # Run model inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=512,  # Allow longer responses for multiple HOI
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

        # Extract predicted boxes
        output_ids = outputs.sequences
        pred_boxes_all = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()

        # Smart filtering: Remove giant boxes BEFORE model sees them
        MAX_BOX_AREA_RATIO = 0.5
        box_areas = (pred_boxes_all[:, 2] - pred_boxes_all[:, 0]) * (pred_boxes_all[:, 3] - pred_boxes_all[:, 1])
        valid_box_mask = box_areas < MAX_BOX_AREA_RATIO
        num_filtered = (~valid_box_mask).sum().item()

        if show_verbose and num_filtered > 0:
            print(f"  Filtered {num_filtered} giant boxes (area > {MAX_BOX_AREA_RATIO})")

        # Parse response
        input_token_len = input_ids.shape[1]
        generated_text = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=False)

        if show_verbose:
            print(f"  Response: {generated_text[:150]}...")

        # Extract entities and region indices
        entities, region_indices = parse_hoi_response(generated_text)

        # Deduplicate
        region_indices, entities = deduplicate_boxes(region_indices, entities)

        if show_verbose:
            print(f"  Parsed entities: {entities}")
            print(f"  Region indices: {region_indices}")

        # Filter valid indices
        region_indices_valid = [idx for idx in region_indices if idx < len(pred_boxes_all)]

        if len(region_indices_valid) == 0:
            print(f"  ❌ FAILED - No valid boxes")
            per_image_results.append({
                'image_id': img_id,
                'file_name': file_name,
                'status': 'FAILED - No valid boxes',
                'generated_text': generated_text[:200]
            })
            continue

        # Get selected boxes
        selected_boxes = pred_boxes_all[region_indices_valid]
        selected_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xywh')
        selected_boxes = rescale_box(selected_boxes, img_shape).tolist()

        # Match to ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            selected_boxes, entities, gt_boxes, gt_categories
        )

        if show_verbose:
            print(f"  Matched: {len(matches)}/{len(gt_boxes)} ({len(matches)/len(gt_boxes)*100:.1f}%)")

        # Store prediction
        scores = [1.0] * len(selected_boxes)
        predictions.append({
            'img_id': img_id,
            'entities': entities,
            'boxes': selected_boxes,
            'scores': scores
        })

        per_image_results.append({
            'image_id': img_id,
            'file_name': file_name,
            'num_gt_boxes': len(gt_boxes),
            'num_pred_boxes': len(selected_boxes),
            'num_matched': len(matches),
            'status': 'SUCCESS',
            'generated_text': generated_text[:200]
        })

        # Log per-image metrics to WandB
        if use_wandb:
            wandb.log({
                'image_idx': idx,
                'recall': len(matches) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0,
                'num_predictions': len(selected_boxes),
                'num_gt': len(gt_boxes),
                'num_matched': len(matches),
                'num_unmatched_preds': len(unmatched_preds),
                'num_unmatched_gts': len(unmatched_gts),
                'num_filtered_giant_boxes': num_filtered,
            })

        # Create visualization
        if show_verbose and viz_output_dir is not None:
            viz_path = os.path.join(viz_output_dir, f"{os.path.splitext(file_name)[0]}_hoi_oneshot.jpg")
            try:
                viz_stats = visualize_oneshot_results(
                    img_path=img_path,
                    all_pred_boxes=pred_boxes_all,
                    pred_boxes=selected_boxes,
                    pred_entities=entities,
                    pred_region_indices=region_indices_valid,
                    gt_boxes=gt_boxes,
                    gt_entities=gt_categories,
                    matches=matches,
                    unmatched_preds=unmatched_preds,
                    unmatched_gts=unmatched_gts,
                    output_path=viz_path,
                    img_shape=img_shape,
                    generated_text=generated_text
                )

                # Aggregate stats
                for key in viz_stats:
                    aggregate_stats[key] += viz_stats[key]

                print(f"  💾 Visualization saved to: {viz_path}")

                # Upload visualization to WandB
                if use_wandb:
                    try:
                        wandb.log({
                            f"visualizations/{file_name}": wandb.Image(
                                viz_path,
                                caption=f"Image {idx+1}: {len(matches)}/{len(gt_boxes)} matched | {generated_text[:100]}"
                            )
                        })
                    except Exception as e:
                        print(f"  ⚠️  WandB visualization upload failed: {e}")
            except Exception as e:
                print(f"  ⚠️  Visualization failed: {e}")

    print("\n" + "=" * 70)
    print("Inference complete!")
    print(f"Total images: {len(dataset)}")
    print(f"Valid predictions: {len(predictions)}")

    # Save predictions
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    # Convert to COCO format
    print("\nConverting to COCO format...")
    coco_predictions = convert_to_coco_format(predictions, dataset)
    print(f"Generated {len(coco_predictions)} COCO-format predictions")

    with open(args.result_file, 'w') as f:
        json.dump(coco_predictions, f)

    # Create ground truth
    print("Creating COCO ground truth...")
    gt_coco = create_coco_ground_truth(dataset)

    gt_file = args.result_file.replace('_predictions_', '_ground_truth_')
    with open(gt_file, 'w') as f:
        json.dump(gt_coco, f)

    # Run COCO evaluation
    print("\n" + "=" * 70)
    print("Running COCO evaluation...")
    print("=" * 70)

    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_predictions) if len(coco_predictions) > 0 else coco_gt

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = [1, 10, 500]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        'AR': coco_eval.stats[8],
        'AR@0.5': compute_ar_at_iou(coco_eval, 0.5),
        'AR@0.75': compute_ar_at_iou(coco_eval, 0.75),
        'ARs': coco_eval.stats[9],
        'ARm': coco_eval.stats[10],
        'ARl': coco_eval.stats[11]
    }

    # Print results
    print("\n" + "=" * 70)
    print("SWIG-HOI One-Shot HOI Grounding Metrics")
    print("=" * 70)
    print(f"{'Metric':<12} {'Value':>8}  {'Description':<40}")
    print("-" * 70)
    print(f"{'AR':<12} {metrics['AR']*100:>7.1f}%  {'Average Recall @ IoU=0.50:0.95':<40}")
    print(f"{'AR@0.5':<12} {metrics['AR@0.5']*100:>7.1f}%  {'Average Recall @ IoU=0.50':<40}")
    print(f"{'AR@0.75':<12} {metrics['AR@0.75']*100:>7.1f}%  {'Average Recall @ IoU=0.75':<40}")
    print(f"{'AR@s':<12} {metrics['ARs']*100:>7.1f}%  {'Small objects':<40}")
    print(f"{'AR@m':<12} {metrics['ARm']*100:>7.1f}%  {'Medium objects':<40}")
    print(f"{'AR@l':<12} {metrics['ARl']*100:>7.1f}%  {'Large objects':<40}")
    print("=" * 70)

    # Save metrics
    metrics_file = args.result_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Log final summary to WandB
    if use_wandb:
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value (%)", "Description"],
            data=[
                ["AR", metrics['AR'] * 100, "Average Recall @ IoU=0.50:0.95"],
                ["AR@0.5", metrics['AR@0.5'] * 100, "Average Recall @ IoU=0.50"],
                ["AR@0.75", metrics['AR@0.75'] * 100, "Average Recall @ IoU=0.75"],
                ["ARs", metrics['ARs'] * 100, "Small objects (area < 32²)"],
                ["ARm", metrics['ARm'] * 100, "Medium objects (32² < area < 96²)"],
                ["ARl", metrics['ARl'] * 100, "Large objects (area > 96²)"],
            ]
        )

        # Log final metrics and summary
        wandb.log({
            "final_metrics/table": summary_table,
            "final_metrics/AR": metrics['AR'],
            "final_metrics/AR@0.5": metrics['AR@0.5'],
            "final_metrics/AR@0.75": metrics['AR@0.75'],
            "final_metrics/ARs": metrics['ARs'],
            "final_metrics/ARm": metrics['ARm'],
            "final_metrics/ARl": metrics['ARl'],
        })

        # Set summary statistics (visible in WandB overview)
        wandb.run.summary.update({
            "AR": metrics['AR'],
            "AR@0.5": metrics['AR@0.5'],
            "AR@0.75": metrics['AR@0.75'],
            "ARs": metrics['ARs'],
            "ARm": metrics['ARm'],
            "ARl": metrics['ARl'],
            "total_images": len(dataset),
            "total_predictions": len(predictions),
            "predictions_file": args.result_file,
            "metrics_file": metrics_file,
        })

        # Finish WandB run
        wandb.finish()
        print("\n✓ Results logged to Weights & Biases")

    return metrics


def compute_ar_at_iou(coco_eval, iou_thr):
    """Compute Average Recall at specific IoU threshold"""
    iou_thresholds = coco_eval.params.iouThrs
    if iou_thr in iou_thresholds:
        iou_idx = list(iou_thresholds).index(iou_thr)
    else:
        return 0.0

    recalls = coco_eval.eval['recall']
    if recalls is None or recalls.size == 0:
        return 0.0

    recall_at_iou = recalls[iou_idx, :, 0, 2]  # All categories, all areas, maxDets=500
    valid_recalls = recall_at_iou[recall_at_iou > -1]
    ar = valid_recalls.mean() if len(valid_recalls) > 0 else 0.0

    return float(ar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWIG-HOI One-Shot HOI Grounding Evaluation")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--img-prefix", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--box_score_thres", type=float, default=0.15)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="swig-hoi-oneshot-eval", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name (auto-generated if not provided)")

    args = parser.parse_args()

    if args.batch_size_per_gpu != 1:
        print("Warning: batch_size must be 1, setting to 1")
        args.batch_size_per_gpu = 1

    eval_model(args)
