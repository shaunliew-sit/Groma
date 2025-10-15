#!/usr/bin/env python3
"""
HOI-Grounding Evaluation Script

Evaluates model's ability to ground HOI interactions in instruction format.
Tests whether the model can correctly identify and localize person-object pairs
when given queries like "Describe what each person is doing with objects individually."

Metrics:
- Triplet Accuracy (TA): Percentage of correctly grounded HOI triplets at IoU=0.5
  A triplet is correct if BOTH person and object boxes have IoU > 0.5 with ground truth
- Average Recall (AR): Mean recall over IoU thresholds [0.5:0.95:0.05] (LVIS-style)
- AR@0.5: Recall at IoU=0.5
- AR@0.75: Recall at IoU=0.75
- AR@s, AR@m, AR@l: Recall by interaction size (small, medium, large)
- Per-action and per-object breakdown
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from groma.utils import disable_torch_init
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.datasets.hoi_grounding_eval import (
    HOIGroundingDataset,
    extract_hoi_triplets,
    calculate_iou
)


def parse_model_response(response_text):
    """
    Parse model response to extract predicted triplets.

    Expected format:
    "The <p> person </p> <roi> <ground_box> </roi> is <p> riding </p> the <p> bicycle </p> <roi> <ground_box> </roi>."

    Returns:
        List of (action, object_name) tuples in order of appearance
    """
    import re

    # Extract all <p> phrases and <ground_box> tokens
    phrase_pattern = r'<p>\s*(.*?)\s*</p>'
    phrases = re.findall(phrase_pattern, response_text)

    # Count <ground_box> tokens
    ground_box_count = response_text.count('<ground_box>')

    # Parse triplets: person + action + object (2 ground_box tokens per triplet)
    triplets = []
    if ground_box_count % 2 == 0 and len(phrases) >= 3:
        # Process triplets: skip first "person", then read action + object pairs
        for i in range(1, len(phrases) - 1, 2):
            if i + 1 < len(phrases):
                action = phrases[i].strip()
                obj = phrases[i + 1].strip()
                triplets.append((action, obj))

    return triplets


def parse_predicted_hoi_triplets(response_text, box_idx_token_ids, pred_boxes_dict):
    """
    Parse model response to extract HOI triplets based on text structure.

    The response follows pattern:
    "The <p> person </p> <roi> <r#> </roi> is <p> action </p> the <p> object </p> <roi> <r#> </roi>"

    We need to extract pairs of (person_box_idx, object_box_idx) from the text.

    Args:
        response_text: Model's generated text with <r#> tokens
        box_idx_token_ids: List of token IDs for <r0>, <r1>, ...
        pred_boxes_dict: Dict mapping box index to actual box coordinates

    Returns:
        List of (person_box, object_box) tuples
    """
    import re

    # Pattern to match: <p> phrase </p> <roi> <r#> </roi>
    # We look for sequences of person...action...object with their box indices
    pattern = r'<p>\s*(.*?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'
    matches = re.findall(pattern, response_text)

    if not matches:
        return []

    triplets = []
    i = 0
    while i < len(matches):
        # Look for pattern: person <r#> ... action ... object <r#>
        if i + 2 < len(matches):
            phrase1, box_idx1 = matches[i]
            phrase2, box_idx2 = matches[i + 1]  # This should be action
            phrase3, box_idx3 = matches[i + 2]  # This should be object

            # Check if phrase1 contains "person" and we have boxes
            if 'person' in phrase1.lower():
                person_idx = int(box_idx1)
                # phrase2 is usually the action, phrase3 is object
                # The object's box is in box_idx3
                object_idx = int(box_idx3)

                if person_idx in pred_boxes_dict and object_idx in pred_boxes_dict:
                    triplets.append((pred_boxes_dict[person_idx], pred_boxes_dict[object_idx]))

                # Move to next triplet (skip person, action, object)
                i += 3
            else:
                i += 1
        else:
            break

    return triplets


def match_predicted_to_gt_triplets(pred_triplets, gt_triplets, iou_threshold=0.5):
    """
    Match predicted HOI triplets to ground truth triplets.

    Args:
        pred_triplets: List of (person_box, object_box) tuples from parsed response
        gt_triplets: List of (person_box, object_box) tuples from ground truth
        iou_threshold: IoU threshold for box matching (default: 0.5)

    Returns:
        matched_count: Number of correctly matched triplets
        total_gt: Total ground truth triplets
        total_pred: Total predicted triplets
    """
    if len(pred_triplets) == 0:
        return 0, len(gt_triplets), 0

    matched = 0

    # For each ground truth triplet, try to find a matching predicted triplet
    for gt_person, gt_object in gt_triplets:
        for pred_person, pred_object in pred_triplets:
            # Both person and object boxes must match
            person_iou = calculate_iou(gt_person, pred_person)
            object_iou = calculate_iou(gt_object, pred_object)

            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                matched += 1
                break  # Found a match for this GT triplet

    return matched, len(gt_triplets), len(pred_triplets)


def compute_recall_at_iou(pred_triplets, gt_triplets, iou_threshold):
    """
    Compute recall at a specific IoU threshold.

    Args:
        pred_triplets: List of (person_box, object_box) tuples
        gt_triplets: List of (person_box, object_box) tuples
        iou_threshold: IoU threshold for matching

    Returns:
        recall: Recall value (matched / total_gt)
    """
    if len(gt_triplets) == 0:
        return 0.0

    if len(pred_triplets) == 0:
        return 0.0

    matched = 0
    for gt_person, gt_object in gt_triplets:
        for pred_person, pred_object in pred_triplets:
            person_iou = calculate_iou(gt_person, pred_person)
            object_iou = calculate_iou(gt_object, pred_object)

            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                matched += 1
                break

    return matched / len(gt_triplets)


def compute_interaction_area(person_box, object_box):
    """
    Compute the area of the union bounding box covering both person and object.

    Args:
        person_box: [x, y, w, h] format
        object_box: [x, y, w, h] format

    Returns:
        area: Area of the union bounding box
    """
    # Convert to xyxy format
    p_x1, p_y1 = person_box[0], person_box[1]
    p_x2, p_y2 = person_box[0] + person_box[2], person_box[1] + person_box[3]

    o_x1, o_y1 = object_box[0], object_box[1]
    o_x2, o_y2 = object_box[0] + object_box[2], object_box[1] + object_box[3]

    # Union bounding box
    union_x1 = min(p_x1, o_x1)
    union_y1 = min(p_y1, o_y1)
    union_x2 = max(p_x2, o_x2)
    union_y2 = max(p_y2, o_y2)

    union_w = union_x2 - union_x1
    union_h = union_y2 - union_y1

    return union_w * union_h


def categorize_by_size(triplet_area):
    """
    Categorize interaction by size (COCO/LVIS standard).

    Args:
        triplet_area: Area of the interaction union box

    Returns:
        'small', 'medium', or 'large'
    """
    # COCO/LVIS size thresholds
    if triplet_area < 32 ** 2:  # < 1024
        return 'small'
    elif triplet_area < 96 ** 2:  # < 9216
        return 'medium'
    else:
        return 'large'


def rescale_boxes_to_absolute(boxes, img_shape):
    """
    Rescale normalized boxes [0, 1] to absolute pixel coordinates.

    Args:
        boxes: Tensor of shape [N, 4] in normalized [cx, cy, w, h] format
        img_shape: (height, width) tuple

    Returns:
        List of boxes in absolute [x, y, w, h] format
    """
    import torchvision
    h, w = img_shape

    # Convert from cxcywh to xywh
    boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xywh')

    # Scale to absolute coordinates
    boxes[:, 0] = boxes[:, 0] * w  # x
    boxes[:, 1] = boxes[:, 1] * h  # y
    boxes[:, 2] = boxes[:, 2] * w  # width
    boxes[:, 3] = boxes[:, 3] * h  # height

    return boxes.tolist()


def collate_fn(batch):
    """Custom collate function for batch_size=1 evaluation"""
    assert len(batch) == 1, "HOI grounding evaluation only supports batch_size=1"
    return batch[0]


def eval_model(args):
    """Main evaluation function"""
    print("=" * 80)
    print("HOI-Grounding Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Instruction file: {args.instruction_file}")
    print(f"Image prefix: {args.img_prefix}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Box score threshold: {args.box_score_thres}")
    print(f"Use [grounding] prefix: {args.use_grounding_prefix}")
    print("=" * 80)

    # Initialize model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f"\nLoading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres
    model.eval()
    print("✓ Model loaded")

    # Initialize dataset (uses mmdet preprocessing like LVIS)
    print(f"\nLoading dataset...")
    dataset = HOIGroundingDataset(
        instruction_file=args.instruction_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='llava'
    )
    print(f"✓ Dataset loaded: {len(dataset)} examples")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # Set to 0 for debugging, can increase later
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluation metrics
    total_matched = 0
    total_gt_triplets = 0
    total_pred_triplets = 0
    total_images = 0
    invalid_responses = 0

    # AR metrics: Track recalls at multiple IoU thresholds
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # [0.5, 0.55, ..., 0.95]
    recalls_at_iou = []  # Store recall for each image at each IoU threshold

    # AR by size metrics
    recalls_by_size = {
        'small': [],
        'medium': [],
        'large': []
    }

    # Detailed metrics
    per_action_stats = defaultdict(lambda: {"matched": 0, "total": 0})
    per_object_stats = defaultdict(lambda: {"matched": 0, "total": 0})

    results = []

    print(f"\nStarting evaluation...")
    print(f"Max images to process: {args.max_images if args.max_images > 0 else 'All'}")

    for idx, data_item in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.max_images > 0 and idx >= args.max_images:
            break

        image = data_item['image']  # Already preprocessed
        query = data_item['query']
        gt_boxes = data_item['gt_boxes']
        gt_box_inds = data_item['gt_box_inds']
        gt_response = data_item['gt_response']
        file_name = data_item['file_name']
        width = data_item['width']
        height = data_item['height']

        # Extract ground truth triplets
        gt_triplets = extract_hoi_triplets(gt_boxes, gt_box_inds)

        if len(gt_triplets) == 0:
            continue

        # Conditionally add [grounding] prefix based on argument
        if args.use_grounding_prefix:
            if not query.startswith('[grounding]'):
                query = '[grounding] ' + query
        else:
            # Remove [grounding] prefix if present
            if query.startswith('[grounding]'):
                query = query.replace('[grounding]', '').strip()

        # Image is already preprocessed
        image_processed = image.unsqueeze(0).cuda()

        # Prepare conversation prompt
        # Use the same conversation format as LVIS evaluation
        from groma.data.conversation import conv_templates

        conv_temp = conv_templates['llava']
        conversations = []

        # Initial greeting (same as LVIS)
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((conv_temp.roles[0], instruct))
        conversations.append((conv_temp.roles[1], answer))

        # User query about HOI
        conversations.append((conv_temp.roles[0], query))
        conversations.append((conv_temp.roles[1], ''))

        prompt = conv_temp.get_prompt(conversations)

        # Tokenize
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True
        ).input_ids.cuda()

        # Generate response
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_processed,
                use_cache=True,
                do_sample=False,
                max_new_tokens=512,  # Longer for full HOI descriptions
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

        # Extract predicted boxes
        output_ids = outputs.sequences
        pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()

        # Get predicted box indices from generated tokens
        input_token_len = input_ids.shape[1]
        predicted_box_tokens = [
            id for id in output_ids[0, input_token_len:]
            if id in model.box_idx_token_ids
        ]
        selected_box_inds = [
            model.box_idx_token_ids.index(id)
            for id in predicted_box_tokens
        ]
        selected_box_inds = [
            id for id in selected_box_inds
            if id < len(pred_boxes)
        ]

        if len(selected_box_inds) == 0:
            invalid_responses += 1
            results.append({
                'file_name': file_name,
                'gt_triplets': len(gt_triplets),
                'pred_triplets': 0,
                'matched': 0,
                'accuracy': 0.0,
                'status': 'invalid_no_boxes'
            })
            total_gt_triplets += len(gt_triplets)
            total_images += 1
            continue

        # Get selected predicted boxes and rescale
        selected_boxes = pred_boxes[selected_box_inds]
        pred_boxes_absolute = rescale_boxes_to_absolute(selected_boxes, (height, width))

        # Create mapping: box_index -> box_coordinates
        pred_boxes_dict = {}
        for box_idx, box_coords in zip(selected_box_inds, pred_boxes_absolute):
            pred_boxes_dict[box_idx] = box_coords

        # Decode model response text
        response_text = tokenizer.decode(
            output_ids[0, input_token_len:],
            skip_special_tokens=False
        )

        # Parse response to extract HOI triplets based on text structure
        pred_triplets = parse_predicted_hoi_triplets(
            response_text,
            model.box_idx_token_ids,
            pred_boxes_dict
        )

        # Print first 10 responses for debugging
        if idx < 10:
            print(f"\n{'='*80}")
            print(f"Image {idx}: {file_name}")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"\nGT Response:\n{gt_response}")
            print(f"\nPredicted Response:\n{response_text}")
            print(f"\nGT triplets: {len(gt_triplets)}")
            print(f"Predicted triplets (parsed): {len(pred_triplets)}")
            print(f"Unique predicted boxes: {len(pred_boxes_dict)}")
            print(f"{'='*80}\n")

        # Match predicted triplets to ground truth (for TA metric at IoU=0.5)
        matched, total_gt, total_pred = match_predicted_to_gt_triplets(
            pred_triplets,
            gt_triplets,
            iou_threshold=args.iou_threshold
        )

        total_matched += matched
        total_gt_triplets += total_gt
        total_pred_triplets += total_pred
        total_images += 1

        # Compute recalls at multiple IoU thresholds for AR metric
        image_recalls = []
        for iou_thr in iou_thresholds:
            recall = compute_recall_at_iou(pred_triplets, gt_triplets, iou_thr)
            image_recalls.append(recall)
        recalls_at_iou.append(image_recalls)

        # Compute recall by size
        # Categorize GT triplets by size and compute recalls
        gt_by_size = {'small': [], 'medium': [], 'large': []}
        for gt_person, gt_object in gt_triplets:
            area = compute_interaction_area(gt_person, gt_object)
            size_cat = categorize_by_size(area)
            gt_by_size[size_cat].append((gt_person, gt_object))

        for size_cat in ['small', 'medium', 'large']:
            if len(gt_by_size[size_cat]) > 0:
                size_recalls = []
                for iou_thr in iou_thresholds:
                    recall = compute_recall_at_iou(pred_triplets, gt_by_size[size_cat], iou_thr)
                    size_recalls.append(recall)
                recalls_by_size[size_cat].append(size_recalls)

        # Parse ground truth response to extract actions and objects
        gt_parsed = parse_model_response(gt_response)
        for action, obj in gt_parsed:
            per_action_stats[action]["total"] += 1
            per_object_stats[obj]["total"] += 1
            # Rough approximation: if any triplet matched, count it
            if matched > 0:
                per_action_stats[action]["matched"] += matched / total_gt
                per_object_stats[obj]["matched"] += matched / total_gt

        results.append({
            'file_name': file_name,
            'gt_triplets': total_gt,
            'pred_triplets': total_pred,
            'matched': matched,
            'accuracy': matched / total_gt if total_gt > 0 else 0.0,
            'gt_response': gt_response,
            'pred_response': response_text,
            'status': 'valid'
        })

    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total images evaluated: {total_images}")
    print(f"Invalid responses (no boxes): {invalid_responses}")
    print(f"\nGround truth triplets: {total_gt_triplets}")
    print(f"Predicted triplets: {total_pred_triplets}")
    print(f"Matched triplets: {total_matched}")

    # Compute TA (Triplet Accuracy) at IoU=0.5
    triplet_accuracy = (total_matched / total_gt_triplets * 100) if total_gt_triplets > 0 else 0.0
    precision = (total_matched / total_pred_triplets * 100) if total_pred_triplets > 0 else 0.0
    recall = (total_matched / total_gt_triplets * 100) if total_gt_triplets > 0 else 0.0

    # Compute AR (Average Recall) metrics
    # AR: Average over all IoU thresholds [0.5:0.95:0.05]
    recalls_at_iou_array = np.array(recalls_at_iou)  # shape: [num_images, 10]
    ar_per_iou = np.mean(recalls_at_iou_array, axis=0) * 100  # Average over images
    ar = np.mean(ar_per_iou)  # Average over IoU thresholds
    ar_50 = ar_per_iou[0]  # Recall at IoU=0.5
    ar_75 = ar_per_iou[5]  # Recall at IoU=0.75

    # AR by size
    ar_small = 0.0
    ar_medium = 0.0
    ar_large = 0.0

    if len(recalls_by_size['small']) > 0:
        small_array = np.array(recalls_by_size['small'])
        ar_small = np.mean(small_array) * 100

    if len(recalls_by_size['medium']) > 0:
        medium_array = np.array(recalls_by_size['medium'])
        ar_medium = np.mean(medium_array) * 100

    if len(recalls_by_size['large']) > 0:
        large_array = np.array(recalls_by_size['large'])
        ar_large = np.mean(large_array) * 100

    # Print metrics with detailed explanations
    print("\n" + "=" * 80)
    print("METRICS EXPLANATION")
    print("=" * 80)
    print("\n1. Triplet Accuracy (TA) @ IoU=0.5:")
    print("   - Measures if BOTH person AND object boxes are correctly localized")
    print("   - A triplet is correct if person_IoU ≥ 0.5 AND object_IoU ≥ 0.5")
    print("   - TA = (# correctly grounded triplets) / (# total GT triplets)")
    print("   - This is a STRICT metric: requires precise localization of paired boxes")
    print("\n2. Average Recall (AR) - LVIS-style:")
    print("   - AR: Mean recall averaged over IoU thresholds [0.5, 0.55, ..., 0.95]")
    print("   - Measures localization quality across different strictness levels")
    print("   - AR@0.5: Recall at IoU=0.5 (more lenient, accepts rough localization)")
    print("   - AR@0.75: Recall at IoU=0.75 (stricter, requires precise localization)")
    print("   - AR@s/m/l: Recall for small/medium/large interaction regions")
    print("   - Size categories (union box area):")
    print("     * Small: area < 1024 pixels² (32x32)")
    print("     * Medium: 1024 ≤ area < 9216 pixels² (96x96)")
    print("     * Large: area ≥ 9216 pixels²")
    print("\n3. Key Differences:")
    print("   - TA uses a single IoU threshold (0.5)")
    print("   - AR averages over multiple thresholds (more comprehensive)")
    print("   - Both measure paired boxes (person + object), not single objects")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 80)
    print(f"{'Triplet Accuracy (TA)':<30} {triplet_accuracy:>9.2f}%")
    print(f"{'Precision':<30} {precision:>9.2f}%")
    print(f"{'Recall @ IoU=0.5':<30} {recall:>9.2f}%")
    print("-" * 80)
    print(f"{'Average Recall (AR)':<30} {ar:>9.2f}%")
    print(f"{'AR @ IoU=0.5':<30} {ar_50:>9.2f}%")
    print(f"{'AR @ IoU=0.75':<30} {ar_75:>9.2f}%")
    print(f"{'AR @ small':<30} {ar_small:>9.2f}%")
    print(f"{'AR @ medium':<30} {ar_medium:>9.2f}%")
    print(f"{'AR @ large':<30} {ar_large:>9.2f}%")

    # Top actions by difficulty
    print(f"\n{'Top 10 Actions (by total count)':<50}")
    print("-" * 80)
    action_sorted = sorted(
        per_action_stats.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:10]
    for action, stats in action_sorted:
        acc = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        print(f"  {action:<30} Total: {stats['total']:>4}  Acc: {acc:>6.2f}%")

    # Top objects by difficulty
    print(f"\n{'Top 10 Objects (by total count)':<50}")
    print("-" * 80)
    object_sorted = sorted(
        per_object_stats.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:10]
    for obj, stats in object_sorted:
        acc = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        print(f"  {obj:<30} Total: {stats['total']:>4}  Acc: {acc:>6.2f}%")

    # Save detailed results
    output_file = args.result_file
    summary = {
        "evaluation_params": {
            "model": args.model_name,
            "instruction_file": args.instruction_file,
            "iou_threshold": args.iou_threshold,
            "box_score_threshold": args.box_score_thres,
            "use_grounding_prefix": args.use_grounding_prefix,
            "max_images": args.max_images if args.max_images > 0 else len(dataset)
        },
        "metrics": {
            "triplet_accuracy": triplet_accuracy,
            "precision": precision,
            "recall": recall,
            "average_recall": ar,
            "ar_50": ar_50,
            "ar_75": ar_75,
            "ar_small": ar_small,
            "ar_medium": ar_medium,
            "ar_large": ar_large,
            "total_images": total_images,
            "total_gt_triplets": total_gt_triplets,
            "total_pred_triplets": total_pred_triplets,
            "matched_triplets": total_matched,
            "invalid_responses": invalid_responses
        },
        "per_action": {k: v for k, v in per_action_stats.items()},
        "per_object": {k: v for k, v in per_object_stats.items()},
        "detailed_results": results[:100]  # Save first 100 for inspection
    }

    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOI-Grounding Evaluation")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma model checkpoint")
    parser.add_argument("--instruction-file", type=str, required=True,
                        help="Path to HOI grounding instruction JSON file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO-DET images directory")
    parser.add_argument("--result-file", type=str, default="hoi_grounding_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for box matching (default: 0.5)")
    parser.add_argument("--box-score-thres", type=float, default=0.15,
                        help="Box confidence threshold (default: 0.15)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Maximum number of images to evaluate (0 = all)")

    # Add mutually exclusive group for grounding prefix
    grounding_group = parser.add_mutually_exclusive_group()
    grounding_group.add_argument("--use-grounding-prefix", dest="use_grounding_prefix",
                                 action="store_true", default=True,
                                 help="Add [grounding] prefix to queries (default)")
    grounding_group.add_argument("--no-grounding-prefix", dest="use_grounding_prefix",
                                 action="store_false",
                                 help="Do NOT add [grounding] prefix to queries")

    args = parser.parse_args()

    eval_model(args)
