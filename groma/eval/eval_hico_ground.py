"""
HICO-DET Object Grounding Evaluation Script

Evaluates object grounding performance on HICO-DET dataset.
Task: Given object category names, predict bounding boxes for those objects.

Metrics: COCO-style AP/AR (similar to LVIS-Ground evaluation)
"""

import os
import json
import torch
import argparse
import torchvision
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.data.datasets.hico_ground import HICOGroundTest, collate_fn


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


def convert_to_coco_format(predictions, dataset):
    """
    Convert predictions to COCO format for evaluation.

    Args:
        predictions: List of dicts with {img_id, category, boxes}
        dataset: HICOGroundTest dataset instance

    Returns:
        List of COCO-format detection results
    """
    results = []
    cat_name_to_id = dataset.name_to_cat_id

    for pred in predictions:
        img_id = pred['img_id']
        category_name = pred['category']
        boxes = pred['boxes']  # List of [x, y, w, h]
        scores = pred['scores']  # List of confidence scores

        # Get COCO category ID
        if category_name not in cat_name_to_id:
            continue

        cat_id = cat_name_to_id[category_name]

        for box, score in zip(boxes, scores):
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": box,  # [x, y, w, h]
                "score": score
            })

    return results


def create_coco_ground_truth(dataset):
    """
    Create COCO-format ground truth annotations from dataset.

    Args:
        dataset: HICOGroundTest dataset instance

    Returns:
        COCO-format annotation dict
    """
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

        # Add annotations for each category
        for cat_name, boxes in item['category_boxes'].items():
            if cat_name not in cat_name_to_id:
                continue

            cat_id = cat_name_to_id[cat_name]

            for box in boxes:
                # Convert box to [x, y, w, h] if needed
                if len(box) == 4:
                    x, y, w, h = box
                else:
                    continue

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
            "description": "HICO-DET Object Grounding Ground Truth",
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres
    model.eval()

    print(f"Model loaded successfully")
    print(f"Box score threshold: {args.box_score_thres}")

    # Load dataset
    dataset = HICOGroundTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='llava'
    )

    # Limit dataset size if requested
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
    invalid_count = 0
    category_stats = defaultdict(lambda: {'total': 0, 'predicted': 0})

    # Store detailed per-image results
    per_image_results = []

    # Determine if we should show verbose output
    # Auto-enable for small datasets, or use command line arg
    show_verbose = args.verbose or len(dataset) <= 100
    if show_verbose:
        print(f"\n{'='*70}")
        print("VERBOSE MODE: Showing detailed results for each image")
        print(f"{'='*70}\n")

    for idx, (categories, category_boxes, img_id, img_shape, file_name, base_item) in enumerate(tqdm(dataloader, disable=show_verbose)):
        # Load and preprocess image once per image
        img_path = os.path.join(args.img_prefix, file_name)
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert('RGB')
        img = img.resize((448, 448), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)

        # Normalize
        mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
        std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
        img = (img - mean) / std

        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        image = img.unsqueeze(0).cuda()

        if show_verbose:
            print(f"\n[Image {idx+1}/{len(dataset)}] {file_name}")
            print(f"  Image ID: {img_id}")
            print(f"  Categories in image: {categories}")

        # Store results for this image
        image_predictions = {}  # category -> boxes
        image_status = {}  # category -> status
        image_generated_text = {}  # category -> generated text

        # Run separate inference for EACH category
        for category in categories:
            category_stats[category]['total'] += 1

            # Build prompt for this specific category
            input_ids, full_prompt = dataset.build_single_category_prompt(base_item, category)
            input_ids = input_ids.cuda()

            # Run model inference for this category
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image,
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=50,  # Allow multiple boxes
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    generation_config=model.generation_config
                )

            # Extract predicted boxes
            output_ids = outputs.sequences
            pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()

            # Parse box tokens from output
            input_token_len = input_ids.shape[1]
            predicted_box_tokens = [
                id for id in output_ids[0, input_token_len:]
                if id in model.box_idx_token_ids
            ]

            # Map tokens to box indices
            selected_box_inds = [
                model.box_idx_token_ids.index(id)
                for id in predicted_box_tokens
            ]

            # Filter valid indices
            selected_box_inds = [
                idx for idx in selected_box_inds
                if idx < len(pred_boxes)
            ]

            # Decode generated text for logging
            generated_text = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=False)
            image_generated_text[category] = generated_text

            # Process predictions for this category
            if len(selected_box_inds) == 0:
                image_predictions[category] = []
                image_status[category] = 'FAILED - No boxes'

                if show_verbose:
                    print(f"\n  [{category}] ❌ FAILED - No boxes predicted")
                    print(f"    Query: 'Locate all <{category}> in this image'")
                    print(f"    GT boxes: {len(category_boxes[category])}")
                    print(f"    Generated: {generated_text[:100]}...")
            else:
                # Convert boxes from cxcywh to xywh format
                selected_boxes = pred_boxes[selected_box_inds]
                selected_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xywh')

                # Rescale to original image dimensions
                selected_boxes = rescale_box(selected_boxes, img_shape).tolist()

                image_predictions[category] = selected_boxes
                image_status[category] = 'SUCCESS'
                category_stats[category]['predicted'] += 1

                # Store prediction
                scores = [1.0] * len(selected_boxes)
                predictions.append({
                    'img_id': img_id,
                    'category': category,
                    'boxes': selected_boxes,
                    'scores': scores
                })

                if show_verbose:
                    print(f"\n  [{category}] ✓ SUCCESS")
                    print(f"    Query: 'Locate all <{category}> in this image'")
                    print(f"    GT boxes: {len(category_boxes[category])}")
                    print(f"    Predicted: {len(selected_boxes)} boxes")
                    for i, box in enumerate(selected_boxes[:2]):
                        print(f"      Box {i+1}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                    if len(selected_boxes) > 2:
                        print(f"      ... and {len(selected_boxes) - 2} more boxes")

        # Store per-image result
        total_gt_boxes = sum(len(category_boxes[cat]) for cat in categories)
        total_pred_boxes = sum(len(image_predictions[cat]) for cat in categories)

        overall_status = 'SUCCESS' if total_pred_boxes > 0 else 'FAILED - No boxes predicted'
        if total_pred_boxes == 0:
            invalid_count += 1

        image_result = {
            'image_id': img_id,
            'file_name': file_name,
            'img_shape': img_shape,
            'categories_queried': categories,
            'ground_truth_boxes': {cat: category_boxes[cat] for cat in categories},
            'predicted_boxes_per_category': image_predictions,
            'num_gt_boxes': total_gt_boxes,
            'num_pred_boxes': total_pred_boxes,
            'status': overall_status,
            'per_category_status': image_status,
            'generated_text_per_category': {cat: text[:200] for cat, text in image_generated_text.items()}
        }
        per_image_results.append(image_result)

        if show_verbose:
            print(f"\n  Overall: GT={total_gt_boxes} boxes, Predicted={total_pred_boxes} boxes")
            print(f"  {'-'*68}")

    print("\n" + "=" * 70)
    print("Inference complete!")
    print(f"Total images processed: {len(dataset)}")
    print(f"Invalid predictions (no boxes): {invalid_count}")
    print(f"Valid predictions: {len(predictions)}")
    print(f"Success rate: {(len(dataset) - invalid_count) / len(dataset) * 100:.1f}%")

    # Save per-image results
    per_image_file = args.result_file.replace('.json', '_per_image_results.json')
    print(f"\nSaving per-image results to: {per_image_file}")
    with open(per_image_file, 'w') as f:
        json.dump(per_image_results, f, indent=2)

    # Print per-category statistics
    print("\nPer-category prediction statistics:")
    print("-" * 70)
    for category, stats in sorted(category_stats.items()):
        pred_rate = stats['predicted'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {category:20s}: {stats['predicted']:4d}/{stats['total']:4d} ({pred_rate:5.1f}%)")

    # Show sample successful and failed cases
    if not show_verbose:  # Only show samples if we didn't show all
        print("\n" + "=" * 70)
        print("Sample Results (first 5 successful, first 5 failed):")
        print("=" * 70)

        successful = [r for r in per_image_results if r['status'] == 'SUCCESS']
        failed = [r for r in per_image_results if 'FAILED' in r['status']]

        print("\n✓ SUCCESSFUL predictions (first 5):")
        for i, result in enumerate(successful[:5]):
            print(f"\n  [{i+1}] {result['file_name']}")
            print(f"      Categories: {result['categories_queried']}")
            print(f"      GT boxes: {result['num_gt_boxes']}, Predicted: {result['num_pred_boxes']}")
            print(f"      Generated: {result['generated_text'][:80]}...")

        print("\n❌ FAILED predictions (first 5):")
        for i, result in enumerate(failed[:5]):
            print(f"\n  [{i+1}] {result['file_name']}")
            print(f"      Categories: {result['categories_queried']}")
            print(f"      GT boxes: {result['num_gt_boxes']}")
            print(f"      Generated: {result['generated_text'][:80]}...")

        print(f"\n  (See {per_image_file} for complete per-image results)")
        print("=" * 70)

    # Convert to COCO format
    print("\nConverting predictions to COCO format...")
    coco_predictions = convert_to_coco_format(predictions, dataset)
    print(f"Generated {len(coco_predictions)} COCO-format predictions")

    # Save predictions
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    print(f"\nSaving predictions to: {args.result_file}")
    with open(args.result_file, 'w') as f:
        json.dump(coco_predictions, f)

    # Create ground truth in COCO format
    print("Creating COCO ground truth annotations...")
    gt_coco = create_coco_ground_truth(dataset)

    # Save ground truth for reference
    gt_file = args.result_file.replace('_predictions_', '_ground_truth_')
    print(f"Saving ground truth to: {gt_file}")
    with open(gt_file, 'w') as f:
        json.dump(gt_coco, f)

    # Run COCO evaluation
    print("\n" + "=" * 70)
    print("Running COCO evaluation...")
    print("=" * 70)

    # Load ground truth and predictions using COCO API
    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_predictions) if len(coco_predictions) > 0 else coco_gt

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'APs': coco_eval.stats[3],
        'APm': coco_eval.stats[4],
        'APl': coco_eval.stats[5],
        'AR': coco_eval.stats[8],
        'ARs': coco_eval.stats[9],
        'ARm': coco_eval.stats[10],
        'ARl': coco_eval.stats[11]
    }

    # Save metrics to JSON
    metrics_file = args.result_file.replace('.json', '_metrics.json')
    print(f"\nSaving metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET Object Grounding Evaluation")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma model checkpoint")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to HICO grounding instruction JSON")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for prediction results")
    parser.add_argument("--box_score_thres", type=float, default=0.15,
                        help="Box confidence threshold")
    parser.add_argument("--batch_size_per_gpu", type=int, default=1,
                        help="Batch size (must be 1 for evaluation)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-image results during inference")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N images (for testing)")

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size_per_gpu != 1:
        print("Warning: batch_size_per_gpu must be 1 for evaluation, setting to 1")
        args.batch_size_per_gpu = 1

    # Run evaluation
    eval_model(args)
