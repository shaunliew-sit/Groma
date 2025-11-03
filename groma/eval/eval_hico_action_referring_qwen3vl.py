"""
HICO-DET Action Referring Evaluation Script for Qwen3VL

Evaluates Qwen3VL action prediction using METEOR and CIDEr metrics.

Task: Given (person, object) bounding boxes, predict the connecting action.

Key Differences from Groma Evaluation:
- Uses Qwen3VL model instead of Groma
- Different prompt format (no refer_box tokens)
- Direct bbox input to model
- Same metrics: METEOR, CIDEr, BLEU, ROUGE-L

Usage:
    python groma/eval/eval_hico_action_referring_qwen3vl.py \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --img-prefix /path/to/hico/images/test2015 \
        --ann-file groma_data/benchmarks/hico_action_referring_test.json \
        --pred-file results/predictions.json \
        --verbose \
        --max-images 10
"""

import os
import json
import torch
import argparse
import re
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# COCO evaluation imports
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_qwen3vl_model(model_name, device="auto"):
    """
    Load Qwen3VL model.

    Args:
        model_name: Model identifier
        device: Device to load on

    Returns:
        model, processor
    """
    print(f"Loading Qwen3VL model: {model_name}")
    print(f"Target device: {device}")

    # Handle device mapping
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if device == "auto":
        device_map = "auto"
    elif device.startswith("cuda"):
        if ":" in device:
            device_idx = device.split(":")[1]
        else:
            device_idx = "0"

        if cuda_visible is not None:
            device_map = {"": "cuda:0"}
            print(f"  Note: CUDA_VISIBLE_DEVICES={cuda_visible}, using cuda:0 (physical GPU {cuda_visible})")
        else:
            device_map = {"": f"cuda:{device_idx}"}
    else:
        device_map = {"": device}

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    processor = AutoProcessor.from_pretrained(model_name)

    print(f"✓ Model loaded successfully on device: {model.device}\n")
    return model, processor


def build_action_referring_prompt(object_category=None):
    """
    Build prompt for action referring task.

    Args:
        object_category: Object category name

    Returns:
        List of message dicts for Qwen3VL
    """
    if object_category:
        prompt_text = (
            f"Task: Describe what action the person (in the first bounding box) is performing with the {object_category} (in the second bounding box).\n\n"
            f"Instructions:\n"
            f"1. Focus on the interaction between the person and the {object_category}\n"
            f"2. Provide a SHORT action phrase (2-4 words)\n"
            f"3. Use format: [action] [object] (e.g., 'riding bicycle', 'holding cup')\n\n"
            f"Output Format:\n"
            f"Provide ONLY the action phrase (no explanations, no punctuation)."
        )
    else:
        prompt_text = (
            "Task: Describe what action the person (in the first bounding box) is performing with the object (in the second bounding box).\n\n"
            "Instructions:\n"
            "1. Focus on the interaction between the person and the object\n"
            "2. Provide a SHORT action phrase (2-4 words)\n"
            "3. Include both the action verb and the object\n\n"
            "Output Format:\n"
            "Provide ONLY the action phrase (no explanations, no punctuation)."
        )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image_placeholder"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]


def convert_bbox_to_qwen_format(bbox, img_size):
    """
    Convert bounding box to Qwen3VL format [0, 1000].

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        img_size: (width, height)

    Returns:
        bbox in [0, 1000] range
    """
    width, height = img_size
    x1, y1, x2, y2 = bbox

    # Normalize to [0, 1]
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height

    # Scale to [0, 1000]
    return [
        int(x1_norm * 1000),
        int(y1_norm * 1000),
        int(x2_norm * 1000),
        int(y2_norm * 1000)
    ]


def run_inference(model, processor, image_path, person_bbox, object_bbox, object_category=None):
    """
    Run Qwen3VL inference for action referring.

    Args:
        model: Qwen3VL model
        processor: Qwen3VL processor
        image_path: Path to image
        person_bbox: [x1, y1, x2, y2] for person
        object_bbox: [x1, y1, x2, y2] for object
        object_category: Optional object category

    Returns:
        output_text: Generated action phrase
        image: PIL Image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Convert bboxes to Qwen format
    person_bbox_qwen = convert_bbox_to_qwen_format(person_bbox, (img_width, img_height))
    object_bbox_qwen = convert_bbox_to_qwen_format(object_bbox, (img_width, img_height))

    # Build prompt
    messages = build_action_referring_prompt(object_category)

    # Replace placeholder with actual image
    for msg in messages:
        for content in msg["content"]:
            if content.get("type") == "image":
                content["image"] = image

    # Add bbox information to prompt
    text_content = messages[0]["content"][1]["text"]
    bbox_prompt = (
        f"\n\nPerson bounding box: {person_bbox_qwen}\n"
        f"Object bounding box: {object_bbox_qwen}\n\n"
        f"{text_content}"
    )
    messages[0]["content"][1]["text"] = bbox_prompt

    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=None,
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text, image


def clean_action_response(response_text):
    """
    Clean action response to extract action phrase.

    Args:
        response_text: Raw model output

    Returns:
        cleaned action phrase (lowercase)
    """
    response = response_text.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
    ]

    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    # Remove trailing punctuation
    response = response.rstrip('.!?,;:')

    # Convert to lowercase
    response = response.lower().strip()

    return response


def visualize_action_triplet(image_path, person_bbox, object_bbox, predicted_action,
                            gt_action, object_category, output_path):
    """
    Visualize action referring result.

    Args:
        image_path: Path to image
        person_bbox: [x1, y1, x2, y2] for person
        object_bbox: [x1, y1, x2, y2] for object
        predicted_action: Predicted action
        gt_action: Ground truth action
        object_category: Object category
        output_path: Output path
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Try to load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw person bbox (red)
    px1, py1, px2, py2 = person_bbox
    draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    draw.text((px1, py1 - 25), "Person", fill="red", font=font_small)

    # Draw object bbox (blue)
    ox1, oy1, ox2, oy2 = object_bbox
    draw.rectangle([ox1, oy1, ox2, oy2], outline="blue", width=4)
    draw.text((ox1, oy1 - 25), object_category.capitalize(), fill="blue", font=font_small)

    # Draw connecting line (green)
    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    draw.line([person_center, object_center], fill="green", width=3)

    # Add predicted action
    draw.text((10, 10), f"Predicted: {predicted_action}", fill="white", font=font)

    # Add ground truth
    draw.text((10, 40), f"GT: {gt_action}", fill="yellow", font=font)

    # Match indicator
    match = predicted_action.lower().strip() == gt_action.lower().strip()
    match_text = "✓ MATCH" if match else "✗ MISMATCH"
    match_color = "green" if match else "red"
    draw.text((10, 70), match_text, fill=match_color, font=font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def eval_model(args):
    """Main evaluation function"""

    print("=" * 80)
    print("HICO-DET Action Referring Evaluation (Qwen3VL)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Device:      {args.device}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    if args.verbose:
        print(f"Verbose:     ENABLED (per-triplet results + visualizations)")
    print("=" * 80)
    print()

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico_action_qwen3vl_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "dataset": "HICO-DET-Action",
                    "task": "action_referring",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["hico", "action-referring", "qwen3vl"]
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

    # Load model
    model, processor = load_qwen3vl_model(args.model_name, args.device)

    # Load annotations
    print(f"Loading annotations from: {args.ann_file}")
    with open(args.ann_file, 'r') as f:
        data = json.load(f)

    # Check if it's COCO format or regular format
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        # COCO format
        print("Detected COCO format annotation file")
        images_dict = {img['id']: img for img in data['images']}
        annotations_dict = {ann['image_id']: ann for ann in data['annotations']}

        # Convert to our format
        dataset_samples = []
        for img_id, img_info in images_dict.items():
            ann = annotations_dict.get(img_id)
            if ann:
                dataset_samples.append({
                    'file_name': img_info['file_name'],
                    'triplet_id': img_id,
                    'person_bbox': img_info['subject_bbox'],
                    'object_bbox': img_info['object_bbox'],
                    'object_category': img_info.get('object_category', 'object'),
                    'gt_action': ann['caption']
                })
    elif isinstance(data, list):
        # Regular format (list of samples with conversation)
        print("Detected regular format annotation file")
        dataset_samples = []
        for idx, sample in enumerate(data):
            box_inds = sample['conversation'][0]['box_inds']
            boxes = sample['boxes']
            dataset_samples.append({
                'file_name': sample['file_name'],
                'triplet_id': idx,
                'person_bbox': boxes[box_inds[0]],
                'object_bbox': boxes[box_inds[1]],
                'object_category': None,
                'gt_action': sample['conversation'][1]['value']
            })
    else:
        raise ValueError("Unknown annotation format")

    print(f"Loaded {len(dataset_samples)} action referring triplets")

    # Limit dataset if requested
    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"\n⚠️  Limiting evaluation to first {args.max_images} triplets")
        dataset_samples = dataset_samples[:args.max_images]

    print(f"\nDataset: {len(dataset_samples)} triplets")
    print("=" * 80)

    # Storage for results
    predictions = []  # For COCO evaluation
    per_triplet_results = []  # For detailed analysis

    # Per-action statistics
    action_stats = defaultdict(lambda: {
        'total': 0,
        'exact_match': 0,
        'predictions': [],
        'ground_truths': []
    })

    # Visualization setup
    viz_dir = None
    if args.verbose:
        viz_dir = args.pred_file.replace('.json', '_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}\n")

    show_verbose = args.verbose or len(dataset_samples) <= 100

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        file_name = sample['file_name']
        img_path = os.path.join(args.img_prefix, file_name)

        # Get boxes and ground truth
        person_bbox = sample['person_bbox']
        object_bbox = sample['object_bbox']
        gt_action = sample['gt_action']

        # Get object category (may be provided or inferred)
        object_category = sample.get('object_category')

        # If not provided, try to infer from gt_action
        if not object_category or object_category == 'object':
            # For HICO, actions are formatted as "action object"
            parts = gt_action.split()
            if len(parts) >= 2:
                object_category = parts[-1]  # Last word is usually the object
            else:
                object_category = None

        if show_verbose:
            print(f"\n[Triplet {idx+1}/{len(dataset_samples)}] {file_name}")
            print(f"  Person bbox: {person_bbox}")
            print(f"  Object bbox: {object_bbox}")
            print(f"  GT action: {gt_action}")

        # Run inference
        output_text, image = run_inference(
            model, processor, img_path,
            person_bbox, object_bbox,
            object_category
        )

        if show_verbose:
            print(f"  Raw output: {output_text[:100]}...")

        # Clean response
        predicted_action = clean_action_response(output_text)

        if show_verbose:
            print(f"  Predicted: {predicted_action}")

        # Store prediction for COCO evaluation
        predictions.append({
            'image_id': idx,
            'caption': predicted_action
        })

        # Store per-triplet result
        exact_match = predicted_action.lower().strip() == gt_action.lower().strip()
        per_triplet_results.append({
            'triplet_id': idx,
            'file_name': file_name,
            'person_bbox': person_bbox,
            'object_bbox': object_bbox,
            'ground_truth': gt_action,
            'prediction': predicted_action,
            'raw_output': output_text[:200],
            'exact_match': exact_match
        })

        # Update action statistics
        action_stats[gt_action]['total'] += 1
        action_stats[gt_action]['exact_match'] += (1 if exact_match else 0)
        action_stats[gt_action]['predictions'].append(predicted_action)
        action_stats[gt_action]['ground_truths'].append(gt_action)

        # Visualize if verbose
        if viz_dir is not None:
            viz_path = os.path.join(viz_dir, f"{idx:05d}_{file_name}")
            try:
                visualize_action_triplet(
                    img_path, person_bbox, object_bbox,
                    predicted_action, gt_action, object_category or "object",
                    viz_path
                )

                # Log to WandB
                if use_wandb:
                    wandb.log({
                        f"visualization/{idx:04d}": wandb.Image(
                            viz_path,
                            caption=f"{file_name} | Pred: {predicted_action} | GT: {gt_action}"
                        )
                    })
            except Exception as e:
                if show_verbose:
                    print(f"  Warning: Visualization failed: {e}")

        # Log per-sample metrics to WandB
        if use_wandb:
            wandb.log({
                'sample_idx': idx,
                'exact_match': 1 if exact_match else 0,
            })

    # Prepare for COCO evaluation
    print("\n" + "=" * 80)
    print("Computing METEOR and CIDEr metrics...")
    print("=" * 80)

    # Create ground truth annotations in COCO format
    annotations = []
    images_info = []
    for idx, sample in enumerate(dataset_samples):
        images_info.append({'id': idx})
        annotations.append({
            'image_id': idx,
            'caption': sample['gt_action'],
            'id': idx
        })

    # Create temporary files for COCO evaluation
    import tempfile
    gt_coco_format = {
        'info': {
            'description': 'HICO Action Referring Ground Truth',
            'version': '1.0',
            'year': 2025
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'images': images_info,
        'annotations': annotations,
        'type': 'captions'
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_coco_format, f)
        gt_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name

    # Run COCO evaluation
    coco = COCO(gt_file)
    coco_result = coco.loadRes(pred_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # Clean up temp files
    os.unlink(gt_file)
    os.unlink(pred_file)

    # Print results
    print("\n" + "=" * 80)
    print("HICO-DET Action Referring Results (Qwen3VL)")
    print("=" * 80)
    print(f"{'Metric':<15} {'Score':>10}  {'Description':<50}")
    print("-" * 80)

    metrics = {}
    for metric, score in coco_eval.eval.items():
        metrics[metric] = score
        desc = {
            'BLEU_1': 'BLEU-1 (unigram overlap)',
            'BLEU_2': 'BLEU-2 (bigram overlap)',
            'BLEU_3': 'BLEU-3 (trigram overlap)',
            'BLEU_4': 'BLEU-4 (4-gram overlap)',
            'METEOR': 'METEOR (semantic similarity)',
            'ROUGE_L': 'ROUGE-L (longest common subsequence)',
            'CIDEr': 'CIDEr (corpus consensus)',
            'SPICE': 'SPICE (semantic propositional content)'
        }.get(metric, metric)
        print(f"{metric:<15} {score*100:>9.2f}%  {desc:<50}")

    # Add exact match accuracy
    total_triplets = len(per_triplet_results)
    exact_matches = sum(1 for r in per_triplet_results if r['exact_match'])
    exact_match_acc = exact_matches / total_triplets if total_triplets > 0 else 0.0
    metrics['exact_match'] = exact_match_acc

    print("-" * 80)
    print(f"{'Exact Match':<15} {exact_match_acc*100:>9.2f}%  {'Exact string match accuracy':<50}")
    print("=" * 80)

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        wandb.log({
            'total_triplets': total_triplets,
            'exact_matches': exact_matches,
        })

    # Per-action statistics
    print("\n" + "=" * 80)
    print("Per-Action Statistics (Top 20 by frequency)")
    print("=" * 80)
    print(f"{'Action':<30} {'Count':>8} {'Exact Match':>12}")
    print("-" * 80)

    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for action, stats in sorted_actions[:20]:
        accuracy = stats['exact_match'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"{action:<30} {stats['total']:>8} {accuracy*100:>11.1f}%")

    print("=" * 80)

    # Save results
    os.makedirs(os.path.dirname(args.pred_file), exist_ok=True)

    # Save predictions
    print(f"\nSaving predictions to: {args.pred_file}")
    with open(args.pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save metrics
    metrics_file = args.pred_file.replace('.json', '_metrics.json')
    print(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save per-triplet results if verbose
    if args.verbose:
        per_triplet_file = args.pred_file.replace('.json', '_per_triplet.json')
        print(f"Saving per-triplet results to: {per_triplet_file}")
        with open(per_triplet_file, 'w') as f:
            json.dump(per_triplet_results, f, indent=2)

        # Save per-action stats
        per_action_file = args.pred_file.replace('.json', '_per_action.json')
        print(f"Saving per-action stats to: {per_action_file}")
        action_stats_dict = {action: stats for action, stats in action_stats.items()}
        with open(per_action_file, 'w') as f:
            json.dump(action_stats_dict, f, indent=2)

    # Save visualizations summary
    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith('.jpg')])
        print(f"\nVisualizations: {viz_dir}/")
        print(f"  Total images saved: {viz_count}")

    if use_wandb:
        wandb.save(args.pred_file)
        wandb.save(metrics_file)
        if args.verbose:
            wandb.save(per_triplet_file)
            wandb.save(per_action_file)

        # Create action performance table
        action_table_data = []
        for action, stats in sorted_actions[:20]:
            accuracy = stats['exact_match'] / stats['total'] if stats['total'] > 0 else 0.0
            action_table_data.append([
                action,
                stats['total'],
                stats['exact_match'],
                f"{accuracy:.1%}"
            ])

        wandb.log({
            "action_performance_table": wandb.Table(
                columns=["Action", "Total", "Exact Match", "Accuracy"],
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
    parser = argparse.ArgumentParser(description="HICO-DET Action Referring Evaluation with Qwen3VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3VL model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to HICO action referring annotation file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO images directory")
    parser.add_argument("--pred-file", type=str, required=True,
                        help="Output file for predictions")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N triplets (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-triplet results and generate visualizations")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hico-action-referring-qwen3vl",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    eval_model(args)
