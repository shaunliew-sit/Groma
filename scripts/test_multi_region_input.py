#!/usr/bin/env python3
"""
Test script for multi-region referring with Groma model.
Based on GitHub issue #35: https://github.com/FoundationVision/Groma/issues/35

Tests the model's ability to handle 2 bounding boxes as input and generate
action descriptions connecting them (e.g., "person sitting on bench").

REAL EXAMPLE FROM HICO-DET TEST SET:
    Image: HICO_test2015_00000001.jpg (640x427)
    Subject: person at [320, 306, 39, 43] (absolute bbox in xywh format)
    Object: bench at [148, 345, 228, 69]
    Ground truth action: "sit on" (action_id: 87)

Usage with real HICO example:
    python scripts/test_multi_region_input.py \
        --model-name groma-7b-finetune \
        --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
        --subject-bbox "320,306,39,43" \
        --object-bbox "148,345,228,69" \
        --ground-truth "sit on" \
        --output-dir test_output \
        --gpu-id 1

The script automatically:
- Loads the image to detect its dimensions
- Normalizes the bboxes if they appear to be in absolute coordinates (>1.0)
- Handles both absolute and normalized coordinates automatically
"""

import os
import sys
import argparse

# CRITICAL: Set GPU BEFORE importing torch
# Parse --gpu-id argument early
if '--gpu-id' in sys.argv:
    try:
        gpu_idx = sys.argv.index('--gpu-id')
        if gpu_idx + 1 < len(sys.argv):
            gpu_id = sys.argv[gpu_idx + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f"[Early GPU Setup] Setting CUDA_VISIBLE_DEVICES={gpu_id}")
    except (ValueError, IndexError):
        pass

import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


def parse_bbox(bbox_string):
    """Parse bbox string 'x,y,w,h' to list of floats"""
    try:
        parts = [float(x.strip()) for x in bbox_string.split(',')]
        if len(parts) != 4:
            raise ValueError("Bbox must have exactly 4 values: x,y,w,h")
        return parts
    except Exception as e:
        raise ValueError(f"Invalid bbox format '{bbox_string}': {e}")


def is_normalized(bbox):
    """Check if bbox is already normalized (all values <= 1.0)"""
    return all(val <= 1.0 for val in bbox)


def normalize_bbox(bbox, img_width, img_height):
    """
    Convert absolute bbox [x,y,w,h] to normalized [x,y,w,h]
    where values are in [0,1] range
    """
    x, y, w, h = bbox
    return [
        x / img_width,
        y / img_height,
        w / img_width,
        h / img_height
    ]


def load_image(image_file):
    """Load image from file or URL"""
    if image_file.startswith('http') or image_file.startswith('https'):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def visualize_bboxes(image, subject_bbox, object_bbox, output_path):
    """Draw bboxes on image for visualization (expects normalized coords)"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    width, height = image.size

    # Convert normalized to pixel coordinates
    def to_pixels(bbox):
        x, y, w, h = bbox
        x_px = int(x * width)
        y_px = int(y * height)
        w_px = int(w * width)
        h_px = int(h * height)
        return [x_px, y_px, x_px + w_px, y_px + h_px]

    # Draw subject in red
    subj_px = to_pixels(subject_bbox)
    draw.rectangle(subj_px, outline="red", width=3)
    draw.text((subj_px[0], max(0, subj_px[1] - 20)), "SUBJECT (Person)", fill="red")

    # Draw object in blue
    obj_px = to_pixels(object_bbox)
    draw.rectangle(obj_px, outline="blue", width=3)
    draw.text((obj_px[0], max(0, obj_px[1] - 20)), "OBJECT", fill="blue")

    img_draw.save(output_path)
    return output_path


def test_multi_region(args):
    """Test multi-region referring with 2 bounding boxes"""

    print("=" * 80)
    print("MULTI-REGION REFERRING TEST (HOI ACTION PREDICTION)")
    print("=" * 80)

    # GPU info (already set before torch import)
    print(f"\n[GPU] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if args.gpu_id is not None:
        print(f"[GPU] Requested GPU ID: {args.gpu_id} (now mapped to device 0)")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    print(f"[GPU] CUDA devices visible: {torch.cuda.device_count()}")
    print(f"[GPU] Current device: {torch.cuda.current_device()}")
    print(f"[GPU] Device name: {torch.cuda.get_device_name(0)}")

    # Show memory info
    if torch.cuda.is_available():
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[GPU] Memory: {mem_allocated:.2f}GB allocated / {mem_reserved:.2f}GB reserved / {mem_total:.2f}GB total")

    # Load image first to get dimensions
    print(f"\nLoading image: {args.image_file}")
    if not os.path.exists(args.image_file):
        print(f"ERROR: Image file not found: {args.image_file}")
        return

    raw_image = load_image(args.image_file)
    original_width, original_height = raw_image.size
    print(f"Image size: {original_width}x{original_height}")

    # Parse bboxes
    try:
        subject_bbox = parse_bbox(args.subject_bbox)
        object_bbox = parse_bbox(args.object_bbox)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nInput bboxes:")
    print(f"  Subject: {subject_bbox}")
    print(f"  Object:  {object_bbox}")

    # Auto-detect if normalization is needed
    subject_needs_norm = not is_normalized(subject_bbox)
    object_needs_norm = not is_normalized(object_bbox)

    if subject_needs_norm or object_needs_norm:
        print(f"\n→ Bboxes appear to be in absolute coordinates (values > 1.0)")
        print(f"→ Auto-normalizing using image dimensions {original_width}x{original_height}")

        if subject_needs_norm:
            subject_bbox_orig = subject_bbox.copy()
            subject_bbox = normalize_bbox(subject_bbox, original_width, original_height)
            print(f"  Subject: {subject_bbox_orig} → {[f'{x:.4f}' for x in subject_bbox]}")

        if object_needs_norm:
            object_bbox_orig = object_bbox.copy()
            object_bbox = normalize_bbox(object_bbox, original_width, original_height)
            print(f"  Object:  {object_bbox_orig} → {[f'{x:.4f}' for x in object_bbox]}")
    else:
        print(f"\n→ Bboxes are already normalized (all values <= 1.0)")

    print(f"\nFinal normalized bboxes:")
    print(f"  Subject: {[f'{x:.4f}' for x in subject_bbox]}")
    print(f"  Object:  {[f'{x:.4f}' for x in object_bbox]}")

    # Visualize bboxes if output dir specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        viz_path = os.path.join(args.output_dir, 'input_visualization.jpg')
        visualize_bboxes(raw_image, subject_bbox, object_bbox, viz_path)
        print(f"\n✓ Input visualization saved: {viz_path}")

    # Load model
    print("\nLoading model...")
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Handle quantization
    kwargs = {}
    if args.quant_type == 'fp16':
        kwargs['torch_dtype'] = torch.float16
    elif args.quant_type == '8bit':
        kwargs['load_in_8bit'] = True
    elif args.quant_type == '4bit':
        int4_quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4'
        )
        kwargs = {'quantization_config': int4_quant_cfg}

    if args.quant_type in ['8bit', '4bit']:
        model = GromaModel.from_pretrained(model_name, **kwargs)
    else:
        model = GromaModel.from_pretrained(model_name, **kwargs).cuda()
    model.init_special_token_id(tokenizer)
    print("✓ Model loaded successfully!")

    # Build conversation
    conversations = []

    # Initial greeting
    instruct = "Here is an image with region crops from it. "
    instruct += f"Image: {DEFAULT_TOKENS['image']}. "
    instruct += f"Regions: {DEFAULT_TOKENS['region']}."
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))

    # Build multi-region query
    # CRITICAL: Use placeholder tokens, NOT actual coordinates
    subject_ref = (DEFAULT_TOKENS['bor'] + DEFAULT_TOKENS['rbox'] +
                   DEFAULT_TOKENS['eor'] + DEFAULT_TOKENS['rfeat'])
    object_ref = (DEFAULT_TOKENS['bor'] + DEFAULT_TOKENS['rbox'] +
                  DEFAULT_TOKENS['eor'] + DEFAULT_TOKENS['rfeat'])

    # Format query with 2 region placeholders
    query = args.query_template.format(subject_ref, object_ref)

    conversations.append((conv_templates['llava'].roles[0], query))
    conversations.append((conv_templates['llava'].roles[1], ''))

    prompt = conv_templates['llava'].get_prompt(conversations)

    print("\n" + "=" * 80)
    print("QUERY SENT TO MODEL:")
    print(query)
    print("=" * 80)

    # Tokenize
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Resize to 448x448 (required by Groma)
    processed_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'].to('cuda')

    # Prepare bboxes tensor
    # CRITICAL: Pass bboxes via refer_boxes parameter
    bboxes = torch.tensor([subject_bbox, object_bbox], dtype=torch.float32).cuda()
    print(f"\nPrepared tensor for model:")
    print(f"  Shape: {bboxes.shape}  # Expected: (2, 4)")
    print(f"  Subject: {bboxes[0].cpu().tolist()}")
    print(f"  Object:  {bboxes[1].cpu().tolist()}")

    # Generate
    print("\nGenerating model response...")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            outputs = model.generate(
                input_ids,
                images=image,
                refer_boxes=[bboxes],  # CRITICAL: Pass 2 bboxes here
                use_cache=True,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

    # Decode response
    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]

    # Check for consistency
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids differ from input_ids')

    # Get raw response
    response_raw = tokenizer.batch_decode(
        output_ids[:, input_token_len:],
        skip_special_tokens=False
    )[0].strip()

    # Get clean response (no special tokens)
    response_clean = tokenizer.batch_decode(
        output_ids[:, input_token_len:],
        skip_special_tokens=True
    )[0].strip()

    # Print results
    print("\n" + "=" * 80)
    print("MODEL RESPONSE (with special tokens):")
    print("=" * 80)
    print(response_raw)
    print("\n" + "=" * 80)
    print("MODEL RESPONSE (clean):")
    print("=" * 80)
    print(response_clean)
    print("=" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Model: {model_name}")
    print(f"✓ Image: {os.path.basename(args.image_file)} ({original_width}x{original_height})")
    print(f"✓ Generated {len(response_clean.split())} words")
    print(f"\n  **Predicted Action**: \"{response_clean}\"")

    if args.ground_truth:
        print(f"  **Ground Truth**:     \"{args.ground_truth}\"")

        # Simple comparison
        pred_lower = response_clean.lower().strip()
        gt_lower = args.ground_truth.lower().strip()
        if pred_lower == gt_lower:
            print(f"  ✓ EXACT MATCH!")
        elif gt_lower in pred_lower or pred_lower in gt_lower:
            print(f"  ~ PARTIAL MATCH")
        else:
            print(f"  ✗ No match")

    print("=" * 80)

    # Save results if output directory specified
    if args.output_dir:
        # Save text results
        result_file = os.path.join(args.output_dir, 'test_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"MULTI-REGION REFERRING TEST RESULTS\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Image: {args.image_file}\n")
            f.write(f"Image size: {original_width}x{original_height}\n\n")
            f.write(f"Subject bbox (normalized): {subject_bbox}\n")
            f.write(f"Object bbox (normalized):  {object_bbox}\n\n")
            f.write(f"Query: {args.query_template}\n\n")
            f.write(f"Model Response (raw):\n{response_raw}\n\n")
            f.write(f"Model Response (clean):\n{response_clean}\n\n")
            if args.ground_truth:
                f.write(f"Ground Truth Action: {args.ground_truth}\n")

        print(f"\n✓ Results saved to: {result_file}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test multi-region referring with 2 bounding boxes (HOI Action)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE (HICO-DET):

Using absolute coordinates (auto-normalized) with GPU selection:
    python scripts/test_multi_region_input.py \
        --model-name checkpoints/groma-finetune/ \
        --image-file /Users/shaunliew/Documents/Intent-Identification-Detection/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
        --subject-bbox "320,306,39,43" \
        --object-bbox "148,345,228,69" \
        --ground-truth "sit on" \
        --output-dir test_output \
        --gpu-id 1

Alternative prompts to try (copy one of these):
    --query-template "Action: {} is _____ {}. Fill in the blank with only the action verb:"
    --query-template "{} is <action> {}. What is <action>? Answer:"
    --query-template "Describe only the action that {} is doing to {}. One phrase:"
    --query-template "What verb describes the interaction? {} and {}:"

The script automatically detects if bboxes are absolute (values > 1.0) and
normalizes them using the image dimensions. No need to specify --normalized flag!

GPU Selection:
- Use --gpu-id 0 to use GPU 0 (physical GPU 0)
- Use --gpu-id 1 to use GPU 1 (physical GPU 1)
- Omit --gpu-id to use default GPU
- Note: Selected GPU becomes "device 0" from PyTorch's perspective (this is normal)

HICO-DET FORMAT:
- Bboxes in annotations are [x, y, w, h] in absolute pixels
- Category_id 1 = person (always the subject)
- Category_id 15 = bench (object in example above)
- Action category_id 87 = "sit_on" → "sit on" (spaces, not underscores)
        """
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="checkpoints/groma-finetune/",
        help="Path to Groma model checkpoint"
    )
    parser.add_argument(
        "--image-file",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--subject-bbox",
        type=str,
        required=True,
        help="Subject bbox as 'x,y,w,h' (absolute or normalized, auto-detected)"
    )
    parser.add_argument(
        "--object-bbox",
        type=str,
        required=True,
        help="Object bbox as 'x,y,w,h' (absolute or normalized, auto-detected)"
    )
    parser.add_argument(
        "--query-template",
        type=str,
        default="Action: {} is _____ {}. Fill in the blank with only the action verb or phrase (e.g., 'sitting on', 'holding', 'riding'):",
        help="Query template with 2 placeholders for subject and object"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default='none',
        choices=['none', 'fp16', '8bit', '4bit'],
        help="Quantization type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save test results and visualizations"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Ground truth action for comparison (e.g., 'sit on')"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU ID to use (e.g., 0, 1, 2). If not specified, uses default GPU."
    )

    args = parser.parse_args()
    test_multi_region(args)
