"""
[GROMA-QWEN V3] Action Referring Evaluation Script (Supports V2 and V3 models)
Part of: Groma Qwen3VL with Interaction Token Architecture

Evaluates Groma Qwen V2/V3 action prediction using METEOR and CIDEr metrics.
Supports both HICO-DET and SWIG-HOI datasets.

Key Features:
- Auto-detects V2 vs V3 models from config.json
- For V3: Adds interaction box (union of person + object) to prompt
- For V2: Uses standard 2-box format
- Same evaluation metrics (METEOR, CIDEr, BLEU, ROUGE-L)

Task: Given (person, object) bounding boxes, predict the connecting action.

Usage:
    # V3 model
    python groma/eval/eval_action_referring_groma_qwen_v3.py \
        --model-name checkpoints/groma-qwen-v3-referring \
        --img-prefix ../data/hico_20160224_det/images/test2015 \
        --ann-file groma_data/benchmarks/hico_action_referring_test.json \
        --pred-file results/predictions_v3.json \
        --verbose

    # V2 model (auto-detected, uses 2-box format)
    python groma/eval/eval_action_referring_groma_qwen_v3.py \
        --model-name checkpoints/groma-qwen-v2-stage2 \
        --img-prefix ../data/hico_20160224_det/images/test2015 \
        --ann-file groma_data/benchmarks/hico_action_referring_test.json \
        --pred-file results/predictions_v2.json
"""

import os
import json
import re
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import Image

from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from groma.model.groma_qwen import GromaQwenModel, GromaQwenConfig
from groma.model.groma_qwen_interaction import (
    GromaQwenInteractionModel,
    GromaQwenInteractionConfig,
)
from groma.utils import disable_torch_init

# Register custom models
AutoConfig.register("groma_qwen", GromaQwenConfig)
AutoModelForCausalLM.register(GromaQwenConfig, GromaQwenModel)
AutoConfig.register("groma_qwen_interaction", GromaQwenInteractionConfig)
AutoModelForCausalLM.register(GromaQwenInteractionConfig, GromaQwenInteractionModel)

# COCO evaluation imports
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from groma.eval.action_referring_utils import (
    clean_action_response,
    visualize_action_triplet,
    compute_per_action_metrics,
    print_per_action_summary,
    save_per_action_metrics_json
)

# Optional W&B import
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


def compute_interaction_box(person_bbox, object_bbox):
    """Compute interaction box as union of person and object boxes.
    
    Args:
        person_bbox: [x1, y1, x2, y2] in Qwen format [0, 1000]
        object_bbox: [x1, y1, x2, y2] in Qwen format [0, 1000]
    
    Returns:
        Interaction box [x1, y1, x2, y2]
    """
    return [
        min(person_bbox[0], object_bbox[0]),  # x1
        min(person_bbox[1], object_bbox[1]),  # y1
        max(person_bbox[2], object_bbox[2]),  # x2
        max(person_bbox[3], object_bbox[3]),  # y2
    ]


def load_groma_qwen_model(model_name, base_model_name=None, gpu_id=0):
    """Load GromaQwenModel V2 or V3 with auto-detection.
    
    Returns:
        model, processor, tokenizer, is_v3 (bool)
    """
    model_name = os.path.expanduser(model_name)
    
    print("=" * 80)
    print(f"[GROMA-QWEN] Loading model from {model_name}")
    print("=" * 80)

    # Check config to detect V2 vs V3
    config_path = os.path.join(model_name, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    
    model_type = config_dict.get("model_type", "groma_qwen")
    is_v3 = model_type == "groma_qwen_interaction"
    
    if is_v3:
        print(f"✓ Detected V3 model (interaction token enabled)")
        config = GromaQwenInteractionConfig.from_pretrained(model_name, trust_remote_code=True)
        ModelClass = GromaQwenInteractionModel
    else:
        print(f"✓ Detected V2 model (standard 2-box format)")
        config = GromaQwenConfig.from_pretrained(model_name, trust_remote_code=True)
        ModelClass = GromaQwenModel
    
    print(f"  Model type: {model_type}")

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
        device = "cuda:0"
        torch.cuda.set_device(0)
        device_map = {"": device}
        dtype = torch.bfloat16
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        device_map = {"": device}
        dtype = torch.float32
        print("! Using CPU")
    
    model = ModelClass.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    model.init_special_token_id(tokenizer)
    model.eval()
    
    return model, processor, tokenizer, is_v3


def convert_bbox_to_qwen_format(bbox, img_size):
    """Convert bounding box to Qwen3VL format [0, 1000]."""
    width, height = img_size
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / width * 1000),
        int(y1 / height * 1000),
        int(x2 / width * 1000),
        int(y2 / height * 1000)
    ]


def format_bbox_native(bbox):
    """Format bbox in native Qwen3VL format."""
    return f"({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})"


def build_referring_prompt_native(subject_bbox_qwen, object_bbox_qwen, is_v3=False):
    """Build prompt for referring task with native Qwen3VL tokens.
    
    Args:
        subject_bbox_qwen: Person bbox in [0, 1000] format
        object_bbox_qwen: Object bbox in [0, 1000] format
        is_v3: If True, add interaction box to prompt
    
    Returns:
        system_message, user_query
    """
    subject_ref = (
        f"{OBJECT_REF_START}person{OBJECT_REF_END}"
        f"{BOX_START}{format_bbox_native(subject_bbox_qwen)}{BOX_END}"
    )
    object_ref = (
        f"{OBJECT_REF_START}object{OBJECT_REF_END}"
        f"{BOX_START}{format_bbox_native(object_bbox_qwen)}{BOX_END}"
    )
    
    system_message = (
        "You are an expert at understanding human-object interactions in images. "
        "You will be given an image with two regions marked by bounding boxes: "
        "one containing a PERSON and one containing an OBJECT. "
        "Your task is to describe what ACTION the person is performing with/on/to the object."
    )
    
    if is_v3:
        # Add interaction box for V3
        interaction_bbox = compute_interaction_box(subject_bbox_qwen, object_bbox_qwen)
        interaction_ref = (
            f"{OBJECT_REF_START}interaction{OBJECT_REF_END}"
            f"{BOX_START}{format_bbox_native(interaction_bbox)}{BOX_END}"
        )
        
        user_query = (
            f"Action Recognition Task: The first region {subject_ref} contains a PERSON. "
            f"The second region {object_ref} contains an OBJECT. "
            f"The interaction region {interaction_ref} shows where the interaction occurs. "
            f"Describe the action the person is performing with this object. "
            f'Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").'
        )
    else:
        # V2 format (no interaction box)
        user_query = (
            f"Action Recognition Task: The first region {subject_ref} contains a PERSON. "
            f"The second region {object_ref} contains an OBJECT. "
            f"Describe the action the person is performing with this object. "
            f'Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").'
        )
    
    return system_message, user_query


def run_groma_qwen_inference(model, processor, image_path, person_bbox, object_bbox, is_v3=False):
    """Run Groma Qwen V2/V3 inference for referring task."""
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    person_bbox_qwen = convert_bbox_to_qwen_format(person_bbox, (img_width, img_height))
    object_bbox_qwen = convert_bbox_to_qwen_format(object_bbox, (img_width, img_height))

    system_message, user_query = build_referring_prompt_native(
        person_bbox_qwen, object_bbox_qwen, is_v3=is_v3
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_query},
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    # Build refer_boxes
    if is_v3:
        # V3: 3 boxes (person, object, interaction)
        interaction_bbox_qwen = compute_interaction_box(person_bbox_qwen, object_bbox_qwen)
        refer_boxes_tensor = torch.tensor(
            [[person_bbox_qwen, object_bbox_qwen, interaction_bbox_qwen]], 
            dtype=torch.float32
        ).to(model_device)
    else:
        # V2: 2 boxes (person, object)
        refer_boxes_tensor = torch.tensor(
            [[person_bbox_qwen, object_bbox_qwen]], 
            dtype=torch.float32
        ).to(model_device)
    
    refer_boxes = [refer_boxes_tensor[0]]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id == tokenizer.unk_token_id:
        eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            refer_boxes=refer_boxes,
            max_new_tokens=50,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return output_text, image


def convert_xyxy_to_cxcywh(bbox, img_size):
    """Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h] normalized."""
    width, height = img_size
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return [cx / width, cy / height, w / width, h / height]


def eval_model(args):
    """Main evaluation function"""
    
    # Determine dataset name from annotation file
    dataset_name = "hico" if "hico" in args.ann_file.lower() else "swig"
    
    print("=" * 80)
    print(f"[GROMA-QWEN V3] {dataset_name.upper()} Action Referring Evaluation")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Dataset:     {dataset_name.upper()}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images}")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("\nInitializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"groma_qwen_v3_{dataset_name}_{timestamp}",
                config={
                    "model": args.model_name,
                    "base_model": args.base_model_name,
                    "dataset": dataset_name,
                    "ann_file": args.ann_file,
                    "max_images": args.max_images,
                    "verbose": args.verbose,
                },
                tags=["groma-qwen-v3", dataset_name, "action-referring"]
            )
            print(f"✓ Weights & Biases initialized successfully!")
            print(f"  Run URL: {wandb.run.url}")
            print(f"  Project: {wandb.run.project}")
            print(f"  Run name: {wandb.run.name}\n")
        except Exception as e:
            print(f"! W&B initialization failed: {e}")
            print(f"  To use WandB, please run: wandb login")
            use_wandb = False

    # Load model (auto-detect V2/V3)
    model, processor, tokenizer, is_v3 = load_groma_qwen_model(
        args.model_name,
        args.base_model_name
    )
    
    print(f"\n{'='*40}")
    print(f"Model Version: {'V3 (Interaction Token)' if is_v3 else 'V2 (Standard)'}")
    print(f"{'='*40}\n")

    # Load annotations
    print(f"Loading annotations from: {args.ann_file}")
    with open(args.ann_file, 'r') as f:
        data = json.load(f)

    # Parse annotations
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        images_dict = {img['id']: img for img in data['images']}
        annotations_dict = {ann['image_id']: ann for ann in data['annotations']}
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
        dataset_samples = []
        for idx, sample in enumerate(data):
            box_inds = sample['conversation'][0]['box_inds']
            boxes = sample['boxes']
            dataset_samples.append({
                'file_name': sample['file_name'],
                'triplet_id': idx,
                'person_bbox': boxes[box_inds[0]],
                'object_bbox': boxes[box_inds[1]],
                'object_category': sample.get('object_category', None),
                'gt_action': sample['conversation'][1]['value']
            })
    else:
        raise ValueError("Unknown annotation format")

    print(f"Loaded {len(dataset_samples)} triplets")

    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"⚠️  Limiting to first {args.max_images} triplets")
        dataset_samples = dataset_samples[:args.max_images]

    # Storage
    predictions = []
    per_triplet_results = []

    viz_dir = None
    if args.verbose:
        pred_dir = os.path.dirname(args.pred_file) if args.pred_file else "results"
        version_str = "v3" if is_v3 else "v2"
        viz_dir = os.path.join(pred_dir, f"visualizations_{version_str}_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}")

    # Inference
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Generating predictions")):
        triplet_id = sample['triplet_id']
        person_bbox = sample['person_bbox']
        object_bbox = sample['object_bbox']
        gt_action = sample['gt_action']
        object_category = sample.get('object_category', 'object')
        file_name = sample['file_name']
        img_path = os.path.join(args.img_prefix, file_name)

        output_text, image = run_groma_qwen_inference(
            model, processor, img_path, person_bbox, object_bbox, is_v3=is_v3
        )

        pred_clean = clean_action_response(output_text)

        predictions.append({
            "image_id": triplet_id,
            "caption": pred_clean
        })

        if args.verbose:
            per_triplet_results.append({
                "triplet_id": triplet_id,
                "prediction": pred_clean,
                "prediction_raw": output_text,
                "ground_truth": gt_action,
                "person_bbox": person_bbox,
                "object_bbox": object_bbox,
                "object_category": object_category,
                "image_file": file_name,
                "model_version": "v3" if is_v3 else "v2"
            })

            if viz_dir:
                viz_path = os.path.join(viz_dir, f"{str(triplet_id).replace('/', '_')}_action.jpg")
                img_size = image.size
                person_bbox_cxcywh = convert_xyxy_to_cxcywh(person_bbox, img_size)
                object_bbox_cxcywh = convert_xyxy_to_cxcywh(object_bbox, img_size)
                
                visualize_action_triplet(
                    img_path=img_path,
                    person_bbox=person_bbox_cxcywh,
                    object_bbox=object_bbox_cxcywh,
                    pred_action=pred_clean,
                    gt_action=gt_action,
                    output_path=viz_path,
                    person_category="person",
                    object_category=object_category
                )

    print(f"\n✓ Generated {len(predictions)} predictions")

    # Save predictions
    os.makedirs(os.path.dirname(args.pred_file) if os.path.dirname(args.pred_file) else '.', exist_ok=True)
    with open(args.pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Predictions saved to: {args.pred_file}")

    if args.verbose:
        per_triplet_file = args.pred_file.replace('.json', '_per_triplet.json')
        with open(per_triplet_file, 'w') as f:
            json.dump(per_triplet_results, f, indent=2)

    # COCO evaluation
    print(f"\nRunning COCO evaluation (METEOR, CIDEr)...")
    
    coco_gt_data = {
        "info": {"description": f"{dataset_name.upper()} Action Referring V3"},
        "licenses": [],
        "images": [],
        "annotations": []
    }
    for sample in dataset_samples:
        triplet_id = sample['triplet_id']
        coco_gt_data["images"].append({"id": triplet_id, "file_name": sample['file_name']})
        coco_gt_data["annotations"].append({
            "image_id": triplet_id,
            "id": triplet_id,
            "caption": sample['gt_action']
        })
    
    coco_gt_file = args.pred_file.replace('.json', '_coco_gt.json')
    with open(coco_gt_file, 'w') as f:
        json.dump(coco_gt_data, f)
    
    coco_gt = COCO(coco_gt_file)
    coco_pred = coco_gt.loadRes(args.pred_file)
    evaluator = COCOEvalCap(coco_gt, coco_pred)
    evaluator.params['image_id'] = coco_pred.getImgIds()
    evaluator.evaluate()

    # Print results
    print("\n" + "=" * 80)
    print(f"Evaluation Results ({'V3 Interaction Token' if is_v3 else 'V2 Standard'})")
    print("=" * 80)
    for metric, score in evaluator.eval.items():
        if isinstance(score, (int, float)):
            print(f"{metric}: {score:.4f}")
    print("=" * 80)

    # Save metrics
    metrics_file = args.pred_file.replace('.json', '_metrics.json')
    metrics_dict = {
        k: float(v) if isinstance(v, (int, float)) else str(v) 
        for k, v in evaluator.eval.items()
    }
    metrics_dict['model_version'] = 'v3' if is_v3 else 'v2'
    metrics_dict['dataset'] = dataset_name
    metrics_dict['total_samples'] = len(dataset_samples)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")

    # Per-action breakdown
    per_action_stats = None
    if args.verbose and len(per_triplet_results) > 0:
        per_action_stats = compute_per_action_metrics(per_triplet_results, evaluator)
        print_per_action_summary(per_action_stats, top_k=10)
        per_action_file = args.pred_file.replace('.json', '_per_action.json')
        save_per_action_metrics_json(per_action_stats, per_action_file)

    # Log to W&B
    if use_wandb:
        # Log main metrics
        wandb.log(metrics_dict)
        wandb.log({
            'total_triplets': len(predictions),
            'model_version': 'v3' if is_v3 else 'v2',
        })
        
        # Save artifacts
        wandb.save(args.pred_file)
        wandb.save(metrics_file)
        if args.verbose:
            per_triplet_file = args.pred_file.replace('.json', '_per_triplet.json')
            if os.path.exists(per_triplet_file):
                wandb.save(per_triplet_file)
            if per_action_stats:
                per_action_file = args.pred_file.replace('.json', '_per_action.json')
                wandb.save(per_action_file)
        
        # Create action performance table
        if per_action_stats:
            action_table_data = []
            for action, stats in sorted(per_action_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20]:
                action_table_data.append([
                    action,
                    stats['total'],
                    stats['exact_match'],
                    f"{stats['exact_match']/stats['total']:.1%}" if stats['total'] > 0 else "N/A"
                ])
            wandb.log({
                "action_performance_table": wandb.Table(
                    columns=["Action", "Total", "Exact Match", "Accuracy"],
                    data=action_table_data
                )
            })
        
        wandb.finish()
        print("✓ W&B logging complete")

    print("\n✓ Evaluation complete!")
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Referring Evaluation (Groma Qwen V2/V3)")
    parser.add_argument("--model-name", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--base-model-name", type=str, default=None, help="Base model for processor")
    parser.add_argument("--img-prefix", type=str, required=True, help="Images directory")
    parser.add_argument("--ann-file", type=str, required=True, help="Annotation file")
    parser.add_argument("--pred-file", type=str, required=True, help="Output predictions file")
    parser.add_argument("--max-images", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="groma-qwen-v3-eval",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()
    eval_model(args)

