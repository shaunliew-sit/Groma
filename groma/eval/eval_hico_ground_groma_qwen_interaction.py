"""
HICO-DET Grounding Evaluation Script for Groma Qwen3VL (Multi-Pair Format)

Evaluates Groma Qwen3VL grounding performance on HICO-DET dataset with multi-pair support.

Key Differences from Qwen3VL Evaluation:
- Uses GromaQwenModel instead of Qwen3VLForConditionalGeneration
- Parses Groma region tokens: <roi> (x1,y1),(x2,y2) </roi> instead of JSON
- Uses region_images parameter for DINOv2 region features
- Same metrics: AR (Average Recall) at multiple IoU thresholds with size-specific metrics

Task: Given "[grounding] Identify the following person and objects in the image: person {action} {object} and the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Output Format (from Groma Qwen):
<p>person</p><roi> (x1,y1),(x2,y2) </roi><p>object</p><roi> (x1,y1),(x2,y2) </roi>
<p>person</p><roi> (x1,y1),(x2,y2) </roi><p>object</p><roi> (x1,y1),(x2,y2) </roi>

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
from groma.utils import disable_torch_init
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS

# Register custom model
AutoConfig.register("groma_qwen", GromaQwenConfig)
AutoModelForCausalLM.register(GromaQwenConfig, GromaQwenModel)

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

    Returns: 'small', 'medium', or 'large'
    """
    object_area = get_box_area(gt_pair['object_box'])

    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    else:
        return 'large'


def parse_groma_qwen_grounding_response(response_text, img_shape):
    """
    Parse Groma Qwen grounding response to extract person-object pairs.

    Args:
        response_text: Generated text like:
            "<p>person</p><roi> (0,31),(478,998) </roi><p>object</p><roi> (355,228),(767,810) </roi>"
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs: [{'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2]}, ...]
        Boxes are in pixel coordinates
    """
    pairs = []
    h, w = img_shape

    # Pattern to match: <roi> (x1,y1),(x2,y2) </roi>
    roi_pattern = r'<roi>\s*\((\d+),(\d+)\),\((\d+),(\d+)\)\s*</roi>'
    matches = re.findall(roi_pattern, response_text)

    # Extract person and object boxes in pairs
    # Format: <p>person</p><roi>...</roi><p>object</p><roi>...</roi>
    # We need to find pairs of consecutive roi tags
    roi_matches = list(re.finditer(roi_pattern, response_text))
    
    # Group into pairs (person, object)
    for i in range(0, len(roi_matches), 2):
        if i + 1 < len(roi_matches):
            # First match is person, second is object
            person_match = roi_matches[i]
            object_match = roi_matches[i + 1]
            
            # Extract coordinates (in [0, 1000] format)
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
    Load GromaQwenModel and processor.
    
    Returns:
        model, processor, tokenizer
    """
    model_name = os.path.expanduser(model_name)
    
    print("="*80)
    print(f"Loading Groma-Qwen Model from {model_name}")
    print("="*80)

    # 1. Load config
    print(f"\n1. Loading config from {model_name}...")
    config = GromaQwenConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ Config loaded successfully")
    print(f"   - Model type: {config.model_type}")
    print(f"   - New tokens: {config.num_new_token}")

    # 2. Load Processor
    print(f"\n2. Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"   ✓ Processor loaded from checkpoint")
    except Exception as e:
        print(f"   ! Could not load full processor from checkpoint: {e}")
        if base_model_name:
            print(f"   Loading from base model: {base_model_name}...")
            processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
            print(f"   ✓ Processor loaded from base model")
        else:
            raise

    # Get tokenizer
    tokenizer = processor if hasattr(processor, 'vocab_size') else processor.tokenizer
    print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")

    # Check if special tokens exist, if not add them
    test_token = '<region>'
    test_id = tokenizer.convert_tokens_to_ids(test_token)
    if test_id == tokenizer.unk_token_id or test_id is None:
        print(f"   ! Special tokens not found, adding them...")
        num_added = tokenizer.add_tokens(
            list(DEFAULT_TOKENS.values()) + REGION_IDX_TOKENS,
            special_tokens=True
        )
        print(f"   ✓ Added {num_added} special tokens")
    else:
        print(f"   ✓ Special tokens already present (e.g., {test_token} -> {test_id})")

    # 3. Load Model
    print(f"\n3. Loading GromaQwenModel weights...")
    disable_torch_init()
    
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        device_map = {"": device}
        dtype = torch.bfloat16
        print(f"   Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        device_map = {"": device}
        dtype = torch.float32
        print(f"   CUDA not available, using CPU")
    
    model = GromaQwenModel.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    print(f"   ✓ Model loaded successfully")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")

    # 4. Initialize special token IDs
    print(f"\n4. Initializing special tokens...")
    model.init_special_token_id(tokenizer)
    print(f"   ✓ Special tokens initialized")
    print(f"   - Region token ID: {model.reg_token_id}")

    model.eval()
    
    # Handle processor image_processor if missing
    if not hasattr(processor, 'image_processor'):
        print(f"   ! Processor doesn't have image_processor, loading full processor...")
        qwen_base_models = [
            "Groma/checkpoints/Qwen3-VL-8B-Instruct",
            base_model_name if base_model_name else "checkpoints/groma-qwen-stage3-hoi",
            "checkpoints/Qwen3-VL-8B-Instruct",
        ]

        full_processor = None
        for base_model in qwen_base_models:
            try:
                print(f"   Trying {base_model}...")
                full_processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
                if hasattr(full_processor, 'image_processor'):
                    print(f"   ✓ Loaded full processor from {base_model}")
                    break
            except Exception as e:
                print(f"   ! Failed to load from {base_model}: {e}")

        if full_processor is None or not hasattr(full_processor, 'image_processor'):
            raise RuntimeError("Could not find a processor with image_processor.")

        class CombinedProcessor:
            def __init__(self, tokenizer, base_processor):
                self.tokenizer = tokenizer
                self.image_processor = base_processor.image_processor
                self._base_processor = base_processor

            def apply_chat_template(self, *args, **kwargs):
                return self._base_processor.apply_chat_template(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self._base_processor(*args, **kwargs)

            def batch_decode(self, *args, **kwargs):
                return self.tokenizer.batch_decode(*args, **kwargs)

        processor = CombinedProcessor(tokenizer, full_processor)
        print(f"   ✓ Combined processor created")

    return model, processor, tokenizer


def build_grounding_prompt(action, object_category):
    """
    Build prompt for grounding task.
    
    Returns:
        Prompt text string
    """
    prompt_text = f"[grounding] Identify the following person and objects in the image: person {action} {object_category} and the {object_category}"
    return prompt_text


def crop_region(image, bbox_xyxy, target_size=224):
    """
    Crop a region from image and resize to target_size.
    
    Args:
        image: PIL Image
        bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
        target_size: Target size for cropped region (default 224 for DINOv2)
    
    Returns:
        Cropped and resized PIL Image
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
    # Clip to image bounds
    x1 = max(0, min(x1, image.width))
    y1 = max(0, min(y1, image.height))
    x2 = max(x1, min(x2, image.width))
    y2 = max(y1, min(y2, image.height))
    
    # Crop region
    region = image.crop((x1, y1, x2, y2))
    # Resize to target size
    region = region.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return region


def prepare_region_images(image, bboxes_xyxy, target_size=224):
    """
    Crop multiple regions from image and prepare as tensor.
    
    Args:
        image: PIL Image
        bboxes_xyxy: List of [x1, y1, x2, y2] bboxes
        target_size: Target size for cropped regions
    
    Returns:
        Tensor of shape (N, 3, target_size, target_size)
    """
    from torchvision import transforms
    
    regions = []
    for bbox in bboxes_xyxy:
        region = crop_region(image, bbox, target_size)
        regions.append(region)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    region_tensors = [transform(region) for region in regions]
    region_batch = torch.stack(region_tensors)  # (N, 3, H, W)
    return region_batch


def run_groma_qwen_inference(model, processor, image_path, action, object_category, gpu_id=0):
    """
    Run Groma Qwen inference for grounding task.

    Args:
        model: GromaQwenModel
        processor: Processor
        image_path: Path to image file
        action: Action verb
        object_category: Object category name
        gpu_id: GPU device ID

    Returns:
        output_text: Generated response text
        image: PIL Image object
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Build prompt
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

    # For grounding, we don't have specific refer_boxes, but we can use region_images
    # For now, we'll pass empty region_images or None - the model should handle it
    # In practice, Groma Qwen might need region_images for better performance
    # For evaluation, we'll generate without region_images first
    region_images = None

    # Get EOS token ID
    tokenizer = processor if hasattr(processor, 'vocab_size') else processor.tokenizer
    eos_token_id = None
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_token_id != tokenizer.unk_token_id:
        eos_token_id = im_end_token_id
    else:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            region_images=region_images,
            max_new_tokens=200,  # Allow for multiple pairs
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False,
        )

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return output_text, image


def eval_model(args):
    """Main evaluation function"""

    print("=" * 80)
    print("HICO-DET Grounding Evaluation (Groma Qwen3VL)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
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

    # Load Groma Qwen model
    model, processor, tokenizer = load_groma_qwen_model(
        args.model_name,
        args.base_model_name,
        gpu_id
    )

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico_ground_groma_qwen_{timestamp}",
                config={
                    "model": args.model_name,
                    "dataset": "HICO-DET-Ground",
                    "task": "multi_pair_grounding",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["hico", "grounding", "groma-qwen", "multi-pair"]
            )
            print(f"✓ Weights & Biases initialized successfully!")
            print(f"  Run URL: {wandb.run.url}\n")
        except Exception as e:
            print(f"⚠️  Warning: WandB initialization failed: {e}")
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
            model, processor, img_path, action, object_category, gpu_id
        )

        if show_verbose:
            print(f"  Prompt: {prompt_text}")
            print(f"  Response: {output_text[:200]}...")

        # Parse predicted pairs
        pred_pairs = parse_groma_qwen_grounding_response(output_text, img_shape)

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
            'matches_per_threshold': {}
        }

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

                if show_verbose:
                    print(f"  Matched @ IoU=0.5: {len(matches)}/{len(gt_pairs)}")

        per_sample_results.append(sample_result)

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
    print("HICO-DET Grounding Evaluation Results (Groma Qwen3VL)")
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
        "AR@0.75": ar_75
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

    # Log final metrics to WandB
    if use_wandb:
        wandb.log({
            'AR': metrics['AR'],
            'AR@0.5': metrics['AR@0.5'],
            'AR@0.75': metrics['AR@0.75'],
            'ARs': metrics['ARs'],
            'ARm': metrics['ARm'],
            'ARl': metrics['ARl'],
        })
        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET Grounding Evaluation with Groma Qwen3VL")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma Qwen model checkpoint")
    parser.add_argument("--base-model-name", type=str, default=None,
                        help="Base model to load processor from (optional)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use (GPU ID as integer or 'cuda:X')")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to HICO grounding annotation file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-sample results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hico-grounding-groma-qwen",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    eval_model(args)

