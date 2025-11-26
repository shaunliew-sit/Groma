"""
HICO-DET Action Referring Evaluation Script for Groma Qwen3VL

Evaluates Groma Qwen3VL action prediction using METEOR and CIDEr metrics.

Task: Given (person, object) bounding boxes, predict the connecting action.

Key Differences from Qwen3VL Evaluation:
- Uses GromaQwenModel instead of Qwen3VLForConditionalGeneration
- Uses refer_boxes and region_images parameters
- Parses outputs containing Groma region tokens
- Same metrics: METEOR, CIDEr, BLEU, ROUGE-L

Usage:
    python groma/eval/eval_hico_action_referring_groma_qwen.py \
        --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl/checkpoint-2500 \
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
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from collections import defaultdict

from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from groma.model.groma_qwen import GromaQwenModel, GromaQwenConfig
from groma.utils import disable_torch_init
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS

# Register custom model
AutoConfig.register("groma_qwen", GromaQwenConfig)
AutoModelForCausalLM.register(GromaQwenConfig, GromaQwenModel)

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

    # 2. Load Processor (matching test_groma_qwen_referring.py)
    print(f"\n2. Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"   ✓ Processor loaded from checkpoint")
    except Exception as e:
        print(f"   ! Could not load full processor from checkpoint: {e}")
        # Fallback to base model or Qwen3VL
        fallback_paths = [
            base_model_name,
            "checkpoints/Qwen3-VL-8B-Instruct",
        ]
        fallback_paths = [p for p in fallback_paths if p]
        
        processor = None
        for fallback_path in fallback_paths:
            try:
                print(f"   Loading from {fallback_path}...")
                processor = AutoProcessor.from_pretrained(fallback_path, trust_remote_code=True)
                print(f"   ✓ Processor loaded from {fallback_path}")
                break
            except Exception as e2:
                print(f"   ! Failed to load from {fallback_path}: {type(e2).__name__}")
        
        if processor is None:
            raise RuntimeError("Could not load processor from any available path")

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
        # When CUDA_VISIBLE_DEVICES is set, the visible GPU becomes device 0
        # So we always use device 0 regardless of the original gpu_id
        actual_device_id = 0  # After CUDA_VISIBLE_DEVICES filtering, use device 0
        device = f"cuda:{actual_device_id}"
        torch.cuda.set_device(actual_device_id)
        device_map = {"": device}
        dtype = torch.bfloat16
        print(f"   Using GPU {gpu_id} (mapped to cuda:{actual_device_id}): {torch.cuda.get_device_name(actual_device_id)}")
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
    
    # Handle processor image_processor if missing (matching test_groma_qwen_referring.py)
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


def convert_xyxy_to_cxcywh(bbox, img_size):
    """
    Convert bounding box from [x1, y1, x2, y2] to [cx, cy, w, h] format.
    
    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        img_size: (width, height)
    
    Returns:
        [cx, cy, w, h] in normalized [0, 1] coordinates
    """
    width, height = img_size
    x1, y1, x2, y2 = bbox
    
    # Convert to center format
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    
    # Normalize to [0, 1]
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm = w / width
    h_norm = h / height
    
    return [cx_norm, cy_norm, w_norm, h_norm]


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


def build_referring_prompt(query_template=None, use_new_format=True):
    """
    Build prompt for referring task.
    
    New Qwen3VL-native format (use_new_format=True):
    1. System message: Explicit HOI task description
    2. User: Action Recognition Task with explicit person/object regions
    
    Legacy format (use_new_format=False):
    1. User: "What is ... doing with ...?"
    
    Args:
        query_template: Optional custom query template. If None, uses default.
        use_new_format: If True, use new Qwen3VL-native format. If False, use legacy.
    
    Returns:
        List of messages for Qwen3VL chat template
    """
    # Build subject and object references with SPACES (matching training format)
    # Training format: "<roi> <refer_box> </roi> <refer_feat>" (with spaces)
    subject_ref = (DEFAULT_TOKENS['bor'] + ' ' + DEFAULT_TOKENS['rbox'] + ' ' +
                  DEFAULT_TOKENS['eor'] + ' ' + DEFAULT_TOKENS['rfeat'])
    object_ref = (DEFAULT_TOKENS['bor'] + ' ' + DEFAULT_TOKENS['rbox'] + ' ' +
                 DEFAULT_TOKENS['eor'] + ' ' + DEFAULT_TOKENS['rfeat'])
    
    if use_new_format:
        # New Qwen3VL-native format with explicit task description
        system_message = (
            f"You are an expert at understanding human-object interactions in images. "
            f"You will be given an image with two highlighted regions: one containing a PERSON and one containing an OBJECT. "
            f"Your task is to describe what ACTION the person is performing with/on/to the object. "
            f"Image: {DEFAULT_TOKENS['image']}. Regions: {DEFAULT_TOKENS['region']}."
        )
        
        if query_template:
            query_text = query_template.format(subject_ref, object_ref)
        else:
            query_text = (
                f"Action Recognition Task: The first highlighted region "
                f"{subject_ref} contains a PERSON. "
                f"The second highlighted region "
                f"{object_ref} contains an OBJECT. "
                f"Describe the action the person is performing with this object. "
                f'Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").'
            )
        
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query_text
            }
        ]
    else:
        # Legacy format for backward compatibility
        if query_template:
            query_text = query_template.format(subject_ref, object_ref)
        else:
            query_text = (
                f"What is {subject_ref} doing with {object_ref}? "
                f"Answer with action and target:"
            )
        
        messages = [
            {
                "role": "user",
                "content": query_text
            }
        ]
    
    return messages


def run_groma_qwen_referring_inference(model, processor, image_path, person_bbox, object_bbox, gpu_id=0, use_new_format=True):
    """
    Run Groma Qwen inference for referring task.

    Args:
        model: GromaQwenModel
        processor: Processor
        image_path: Path to image file
        person_bbox: [x1, y1, x2, y2] for person in pixel coordinates
        object_bbox: [x1, y1, x2, y2] for object in pixel coordinates
        gpu_id: GPU device ID
        use_new_format: If True, use new Qwen3VL-native format. If False, use legacy.

    Returns:
        output_text: Generated response text
        image: PIL Image object
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Convert bboxes to Qwen format [0, 1000]
    person_bbox_qwen = convert_bbox_to_qwen_format(person_bbox, (img_width, img_height))
    object_bbox_qwen = convert_bbox_to_qwen_format(object_bbox, (img_width, img_height))

    # Build prompt messages
    messages = build_referring_prompt(use_new_format=use_new_format)
    
    # Add image to the user message (last message in the list)
    # The user message should have the image + query text
    user_message = messages[-1]
    query_text = user_message["content"]  # Save the query text
    user_message["content"] = [
        {"type": "image", "image": image},
        {"type": "text", "text": query_text},
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

    # Prepare refer_boxes: list of tensors, each tensor shape [N, 4] with values in [0, 1000]
    refer_boxes_tensor = torch.tensor(
        [[person_bbox_qwen, object_bbox_qwen]], 
        dtype=torch.float32
    ).to(model_device)
    refer_boxes = [refer_boxes_tensor[0]]  # List of tensors, one per batch item

    # Prepare region_images by cropping regions from the image
    region_images = prepare_region_images(
        image, 
        [person_bbox, object_bbox],
        target_size=224  # DINOv2 input size
    ).to(model_device)

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
            refer_boxes=refer_boxes,
            region_images=region_images,
            max_new_tokens=50,
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
    
    # Determine format to use
    use_new_format = getattr(args, 'use_new_format', True)
    if getattr(args, 'use_legacy_format', False):
        use_new_format = False

    print("=" * 80)
    print("HICO-DET Action Referring Evaluation (Groma Qwen3VL)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    print(f"Format:      {'NEW Qwen3VL-native' if use_new_format else 'LEGACY'}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    if args.verbose:
        print(f"Verbose:     ENABLED (per-triplet results + visualizations)")
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

    # Initialize W&B
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico_action_groma_qwen_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "dataset": "HICO-DET-Action",
                    "task": "action_referring",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["hico", "action-referring", "groma-qwen"]
            )
            print(f"✓ Weights & Biases initialized successfully!")
            print(f"  Run URL: {wandb.run.url}\n")
        except Exception as e:
            print(f"⚠️  Warning: WandB initialization failed: {e}")
            print(f"  Continuing evaluation without WandB logging...\n")
            use_wandb = False

    # Load model
    model, processor, tokenizer = load_groma_qwen_model(
        args.model_name,
        args.base_model_name,
        gpu_id
    )

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
                'object_category': sample.get('object_category', None),
                'gt_action': sample['conversation'][1]['value']
            })
    else:
        raise ValueError("Unknown annotation format")

    print(f"Loaded {len(dataset_samples)} triplets")

    # Limit dataset if requested
    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"\n⚠️  Limiting evaluation to first {args.max_images} triplets")
        dataset_samples = dataset_samples[:args.max_images]

    print(f"\nStarting inference on {len(dataset_samples)} triplets...")
    print("=" * 80)
    print()

    # Storage for results
    predictions = []  # For COCO evaluation
    per_triplet_results = []  # For detailed analysis

    # Visualization setup
    viz_dir = None
    if args.verbose:
        pred_dir = os.path.dirname(args.pred_file) if args.pred_file else "results"
        viz_dir = os.path.join(pred_dir, f"visualizations_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}\n")

    # Inference loop
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Generating action predictions", unit="triplet")):
        triplet_id = sample['triplet_id']
        person_bbox = sample['person_bbox']
        object_bbox = sample['object_bbox']
        gt_action = sample['gt_action']
        object_category = sample.get('object_category', 'object')
        file_name = sample['file_name']
        img_path = os.path.join(args.img_prefix, file_name)

        # Run Groma Qwen inference
        output_text, image = run_groma_qwen_referring_inference(
            model, processor, img_path, person_bbox, object_bbox, gpu_id, use_new_format
        )

        # Get clean prediction (remove Groma tokens)
        pred_clean = clean_action_response(output_text)

        # Store prediction in COCO format
        predictions.append({
            "image_id": triplet_id,  # Triplet ID is treated as "image" ID
            "caption": pred_clean     # Predicted action is treated as "caption"
        })

        # Verbose: store detailed results
        if args.verbose:
            per_triplet_results.append({
                "triplet_id": triplet_id,
                "prediction": pred_clean,
                "prediction_raw": output_text,
                "ground_truth": gt_action,
                "person_bbox": person_bbox,
                "object_bbox": object_bbox,
                "person_category": "person",
                "object_category": object_category,
                "image_file": file_name
            })

            # Visualize triplet (without scores initially, will update after evaluation)
            viz_filename = f"{str(triplet_id).replace('/', '_')}_action.jpg"
            viz_path = os.path.join(viz_dir, viz_filename)

            # Convert bboxes from xyxy to cxcywh for visualization
            image = Image.open(img_path)
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

    print(f"\n✓ Generated predictions for {len(predictions)} triplets\n")

    # Save predictions
    print(f"Saving predictions to: {args.pred_file}")
    os.makedirs(os.path.dirname(args.pred_file) if os.path.dirname(args.pred_file) else '.', exist_ok=True)
    with open(args.pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save per-triplet results (verbose mode)
    if args.verbose:
        per_triplet_file = args.pred_file.replace('.json', '_per_triplet.json')
        with open(per_triplet_file, 'w') as f:
            json.dump(per_triplet_results, f, indent=2)
        print(f"✓ Per-triplet results saved to: {per_triplet_file}")

    # Evaluate with METEOR & CIDEr
    print(f"\nRunning COCO caption evaluation (METEOR, CIDEr, BLEU, ROUGE-L)...")
    print("=" * 80)
    print()

    # Create COCO-format ground truth from dataset_samples
    # This is needed because pycocotools expects dict format, not list
    coco_gt_data = {
        "info": {"description": "HICO-DET Action Referring Evaluation"},
        "licenses": [],
        "images": [],
        "annotations": []
    }
    for sample in dataset_samples:
        triplet_id = sample['triplet_id']
        coco_gt_data["images"].append({
            "id": triplet_id,
            "file_name": sample['file_name']
        })
        coco_gt_data["annotations"].append({
            "image_id": triplet_id,
            "id": triplet_id,
            "caption": sample['gt_action']
        })
    
    # Save temporary COCO-format ground truth
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
    print("Evaluation Results")
    print("=" * 80)
    for metric, score in evaluator.eval.items():
        if isinstance(score, (int, float)):
            print(f"{metric}: {score:.4f}")
    print("=" * 80)

    # Save metrics
    metrics_file = args.pred_file.replace('.json', '_metrics.json')
    metrics_dict = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in evaluator.eval.items()}
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")

    # Log metrics to WandB
    if use_wandb:
        wandb.log(metrics_dict)
        print(f"✓ Metrics logged to W&B\n")

    # Per-action breakdown (verbose mode)
    if args.verbose and len(per_triplet_results) > 0:
        print(f"Computing per-action metrics...")
        per_action_stats = compute_per_action_metrics(per_triplet_results, evaluator)

        # Print summary
        print_per_action_summary(per_action_stats, top_k=10)

        # Save to JSON
        per_action_file = args.pred_file.replace('.json', '_per_action.json')
        save_per_action_metrics_json(per_action_stats, per_action_file)
        print(f"✓ Per-action metrics saved to: {per_action_file}\n")

        # Update per-triplet results with scores and regenerate visualizations
        print(f"Updating per-triplet results with METEOR/CIDEr scores...")
        for result in per_triplet_results:
            triplet_id = result['triplet_id']

            # Get scores for this triplet
            if triplet_id in evaluator.imgToEval:
                scores = evaluator.imgToEval[triplet_id]
                meteor_score = scores['METEOR'] * 100  # Convert to percentage
                cider_score = scores['CIDEr'] * 100
                
                # Add scores to result
                result['meteor_score'] = meteor_score
                result['cider_score'] = cider_score
                
                # Print scores for each triplet
                print(f"  Triplet {triplet_id}: METEOR={meteor_score:.2f}%, CIDEr={cider_score:.2f}%")

                # Regenerate visualization with scores
                img_path = os.path.join(args.img_prefix, result['image_file'])
                viz_filename = f"{str(triplet_id).replace('/', '_')}_action.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)

                # Convert bboxes from xyxy to cxcywh for visualization
                image = Image.open(img_path)
                img_size = image.size
                person_bbox_cxcywh = convert_xyxy_to_cxcywh(result['person_bbox'], img_size)
                object_bbox_cxcywh = convert_xyxy_to_cxcywh(result['object_bbox'], img_size)

                visualize_action_triplet(
                    img_path=img_path,
                    person_bbox=person_bbox_cxcywh,
                    object_bbox=object_bbox_cxcywh,
                    pred_action=result['prediction'],
                    gt_action=result['ground_truth'],
                    output_path=viz_path,
                    person_category=result['person_category'],
                    object_category=result['object_category'],
                    meteor_score=meteor_score,
                    cider_score=cider_score
                )
        
        # Save updated per-triplet results with scores
        per_triplet_file = args.pred_file.replace('.json', '_per_triplet.json')
        with open(per_triplet_file, 'w') as f:
            json.dump(per_triplet_results, f, indent=2)
        print(f"\n✓ Updated per-triplet results with scores saved to: {per_triplet_file}")
        print(f"✓ Visualizations updated with similarity scores\n")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET Action Referring Evaluation with Groma Qwen3VL")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma Qwen model checkpoint")
    parser.add_argument("--base-model-name", type=str, default=None,
                        help="Base model to load processor from (optional)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use (GPU ID as integer or 'cuda:X')")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to HICO images directory")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to HICO action referring annotation file")
    parser.add_argument("--pred-file", type=str, required=True,
                        help="Output file for predictions")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N triplets (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-triplet results with METEOR/CIDEr scores")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hico-action-referring-groma-qwen",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--use-new-format", action="store_true", default=True,
                        help="Use new Qwen3VL-native instruction format (default: True)")
    parser.add_argument("--use-legacy-format", action="store_true",
                        help="Use legacy instruction format (overrides --use-new-format)")

    args = parser.parse_args()

    eval_model(args)

