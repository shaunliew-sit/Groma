"""
SWIG-HOI Grounding Evaluation Script for Qwen3VL (Multi-Pair Format)

Evaluates Qwen3VL grounding performance on SWIG-HOI dataset with multi-pair support.

Key Differences from Groma Evaluation:
- Uses Qwen3VL model instead of Groma
- Parses JSON output instead of region tokens
- Different prompt format (no region tokens)
- Direct bbox prediction in [0, 1000] format
- Same metrics: AR (Average Recall) at multiple IoU thresholds
- Supports person-person interactions

Task: Given "Detect all person-{object} pairs where the person is {action} the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Output Format (from Qwen3VL):
[
  {"pair_id": 1, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]},
  {"pair_id": 2, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}
]

Metrics: Pair-level Precision, Recall, F1 @ IoU thresholds (0.5 to 0.95)
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

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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


def extract_response_from_output(generated_ids, input_ids, processor, is_thinking_model=False):
    """
    Extract response text from model output, handling both thinking and instruct models.

    For thinking models (e.g., Qwen3-VL-8B-Thinking):
        - Response contains two sections: thinking + final answer
        - Sections are separated by </think> token (ID: 151668)
        - Returns both the thinking content and the final answer (JSON output)

    For instruct models (e.g., Qwen3-VL-8B-Instruct):
        - Response contains only the final answer (JSON output directly)
        - Returns empty string for thinking content

    Args:
        generated_ids: Full output token IDs from model.generate() [batch_size, seq_len]
        input_ids: Input token IDs [batch_size, input_seq_len]
        processor: Qwen3VL processor
        is_thinking_model: Whether the model is a thinking model

    Returns:
        tuple: (thinking_content: str, final_answer: str)
            - thinking_content: The reasoning process (empty for instruct models)
            - final_answer: The final response text (JSON for parsing)
    """
    if is_thinking_model:
        # Extract only the tokens after input (full generation)
        output_ids = generated_ids[0][len(input_ids[0]):].tolist()

        # Find </think> token (ID: 151668)
        try:
            think_token_id = 151668
            # Find the last occurrence of </think> token
            index = len(output_ids) - output_ids[::-1].index(think_token_id)

            # Extract thinking content (before </think>)
            thinking_content = processor.decode(output_ids[:index-1], skip_special_tokens=True).strip()

            # Decode only the final answer section (after </think>)
            final_answer = processor.decode(output_ids[index:], skip_special_tokens=True).strip()

            return thinking_content, final_answer
        except ValueError:
            # No </think> token found - try to extract JSON from the response
            full_text = processor.decode(output_ids, skip_special_tokens=True).strip()
            print(f"  ⚠️  Warning: No </think> token found, attempting JSON extraction...")

            # Try to find JSON array in the text (common pattern: [{...}, {...}])
            json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', full_text)
            if json_match:
                print(f"  ✓ Found JSON in malformed output")
                return "", json_match.group(0)

            # No JSON found - return empty array to avoid crash
            print(f"  ⚠️  Warning: No JSON found in thinking model output, returning empty array")
            return "", "[]"
    else:
        # Instruct model: decode entire output (standard behavior)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return "", output_text


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


def parse_qwen3vl_json_response(response_text, img_shape):
    """
    Parse Qwen3VL JSON response to extract person-object pairs.

    Args:
        response_text: Generated JSON text like:
            ```json
            [
              {"pair_id": 1, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]},
              {"pair_id": 2, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}
            ]
            ```
        img_shape: Tuple of (height, width)

    Returns:
        List of pairs: [{'person_box': [x1,y1,x2,y2], 'object_box': [x1,y1,x2,y2]}, ...]
        Boxes are in pixel coordinates
    """
    pairs = []
    h, w = img_shape

    # Clean markdown code fences if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = '\n'.join(lines)

    # Try to parse JSON
    try:
        parsed = json.loads(cleaned_text.strip())
        if isinstance(parsed, list):
            detections = parsed
        elif isinstance(parsed, dict):
            detections = [parsed]
        else:
            return pairs
    except json.JSONDecodeError:
        # Fallback: try regex extraction
        array_pattern = r'\[[\s\S]+\]'
        array_matches = re.findall(array_pattern, cleaned_text)
        if array_matches:
            longest_match = max(array_matches, key=len)
            try:
                parsed = json.loads(longest_match)
                if isinstance(parsed, list):
                    detections = parsed
                else:
                    return pairs
            except:
                return pairs
        else:
            return pairs

    # Convert Qwen3VL format [0, 1000] to pixel coordinates
    for det in detections:
        # Skip non-dict items (e.g., integers or strings in malformed JSON)
        if not isinstance(det, dict):
            continue

        if "person_bbox" in det and "object_bbox" in det:
            person_bbox_qwen = det["person_bbox"]
            object_bbox_qwen = det["object_bbox"]

            if len(person_bbox_qwen) == 4 and len(object_bbox_qwen) == 4:
                # Convert from [0, 1000] to pixel coordinates
                person_box = [
                    (person_bbox_qwen[0] / 1000.0) * w,
                    (person_bbox_qwen[1] / 1000.0) * h,
                    (person_bbox_qwen[2] / 1000.0) * w,
                    (person_bbox_qwen[3] / 1000.0) * h
                ]

                object_box = [
                    (object_bbox_qwen[0] / 1000.0) * w,
                    (object_bbox_qwen[1] / 1000.0) * h,
                    (object_bbox_qwen[2] / 1000.0) * w,
                    (object_bbox_qwen[3] / 1000.0) * h
                ]

                pairs.append({
                    'person_box': person_box,
                    'object_box': object_box,
                    'pair_id': det.get('pair_id', len(pairs))
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


def visualize_qwen3vl_grounding(image_path, pred_pairs, gt_pairs, matches,
                                 action, object_category, iou_threshold=0.5):
    """
    Create visualization comparing predicted pairs vs ground truth pairs.

    Args:
        image_path: Path to image file
        pred_pairs: List of predicted pairs with 'person_box' and 'object_box'
        gt_pairs: List of ground truth pairs
        matches: List of (pred_idx, gt_idx, person_iou, object_iou) from matching
        action: Action verb
        object_category: Object category
        iou_threshold: IoU threshold used for matching

    Returns:
        PIL Image with 3-panel visualization (predictions | ground truth | overlay)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Create 3 copies for 3-panel visualization
    pred_img = image.copy()
    gt_img = image.copy()
    overlay_img = image.copy()

    # Create drawing contexts
    pred_draw = ImageDraw.Draw(pred_img)
    gt_draw = ImageDraw.Draw(gt_img)
    overlay_draw = ImageDraw.Draw(overlay_img)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Color scheme
    color_person_pred = (255, 0, 0)      # Red for predicted person
    color_object_pred = (0, 0, 255)      # Blue for predicted object
    color_person_gt = (0, 255, 0)        # Green for GT person
    color_object_gt = (255, 255, 0)      # Yellow for GT object
    color_matched = (0, 255, 0)          # Green for matched pairs
    color_unmatched = (255, 0, 0)        # Red for unmatched pairs

    # Draw predicted pairs (left panel)
    matched_pred_indices = {m[0] for m in matches}
    for idx, pred_pair in enumerate(pred_pairs):
        is_matched = idx in matched_pred_indices

        # Person box
        person_box = pred_pair['person_box']
        pred_draw.rectangle(person_box, outline=color_person_pred, width=3)
        pred_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}",
                      fill=color_person_pred, font=small_font)

        # Object box
        object_box = pred_pair['object_box']
        pred_draw.rectangle(object_box, outline=color_object_pred, width=3)
        pred_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}",
                      fill=color_object_pred, font=small_font)

        # Draw line connecting pair
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        line_color = color_matched if is_matched else color_unmatched
        pred_draw.line([person_center, object_center], fill=line_color, width=2)

    # Draw ground truth pairs (middle panel)
    matched_gt_indices = {m[1] for m in matches}
    for idx, gt_pair in enumerate(gt_pairs):
        is_matched = idx in matched_gt_indices

        # Person box
        person_box = gt_pair['person_box']
        gt_draw.rectangle(person_box, outline=color_person_gt, width=3)
        gt_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}",
                    fill=color_person_gt, font=small_font)

        # Object box
        object_box = gt_pair['object_box']
        gt_draw.rectangle(object_box, outline=color_object_gt, width=3)
        gt_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}",
                    fill=color_object_gt, font=small_font)

        # Draw line connecting pair
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        line_color = color_matched if is_matched else color_unmatched
        gt_draw.line([person_center, object_center], fill=line_color, width=2)

    # Draw overlay (right panel) - show both predictions and GT with matching
    for match in matches:
        pred_idx, gt_idx, person_iou, object_iou = match
        pred_pair = pred_pairs[pred_idx]
        gt_pair = gt_pairs[gt_idx]

        # Draw GT in green
        overlay_draw.rectangle(gt_pair['person_box'], outline=(0, 255, 0), width=2)
        overlay_draw.rectangle(gt_pair['object_box'], outline=(0, 255, 0), width=2)

        # Draw prediction in blue (dashed effect with thinner line)
        overlay_draw.rectangle(pred_pair['person_box'], outline=(0, 100, 255), width=2)
        overlay_draw.rectangle(pred_pair['object_box'], outline=(0, 100, 255), width=2)

        # Show IoU scores
        avg_iou = (person_iou + object_iou) / 2.0
        overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                         f"IoU:{avg_iou:.2f}", fill=(255, 255, 255), font=small_font)

    # Draw unmatched predictions in red
    for idx in range(len(pred_pairs)):
        if idx not in matched_pred_indices:
            pred_pair = pred_pairs[idx]
            overlay_draw.rectangle(pred_pair['person_box'], outline=(255, 0, 0), width=2)
            overlay_draw.rectangle(pred_pair['object_box'], outline=(255, 0, 0), width=2)
            overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                            "FP", fill=(255, 0, 0), font=small_font)

    # Draw unmatched GT in orange
    for idx in range(len(gt_pairs)):
        if idx not in matched_gt_indices:
            gt_pair = gt_pairs[idx]
            overlay_draw.rectangle(gt_pair['person_box'], outline=(255, 165, 0), width=2)
            overlay_draw.rectangle(gt_pair['object_box'], outline=(255, 165, 0), width=2)
            overlay_draw.text((gt_pair['person_box'][0], gt_pair['person_box'][1] - 18),
                            "FN", fill=(255, 165, 0), font=small_font)

    # Create final 3-panel image
    panel_width = img_width
    panel_height = img_height
    total_width = panel_width * 3
    header_height = 60
    total_height = panel_height + header_height

    final_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    final_draw = ImageDraw.Draw(final_img)

    # Add headers
    title_y = 10
    final_draw.text((panel_width // 2 - 100, title_y), "Predictions", fill=(0, 0, 0), font=font)
    final_draw.text((panel_width + panel_width // 2 - 80, title_y), "Ground Truth", fill=(0, 0, 0), font=font)
    final_draw.text((2 * panel_width + panel_width // 2 - 60, title_y), "Overlay", fill=(0, 0, 0), font=font)

    # Add statistics
    stats_y = 35
    num_matched = len(matches)
    num_fp = len(pred_pairs) - num_matched
    num_fn = len(gt_pairs) - num_matched
    recall = num_matched / len(gt_pairs) if len(gt_pairs) > 0 else 0.0

    stats_text = f"Action: {action} | Object: {object_category} | IoU≥{iou_threshold}"
    final_draw.text((10, stats_y), stats_text, fill=(0, 0, 0), font=small_font)

    metrics_text = f"Pred:{len(pred_pairs)} | GT:{len(gt_pairs)} | Matched:{num_matched} | FP:{num_fp} | FN:{num_fn} | Recall:{recall:.1%}"
    final_draw.text((panel_width + 10, stats_y), metrics_text, fill=(0, 0, 0), font=small_font)

    # Paste images
    final_img.paste(pred_img, (0, header_height))
    final_img.paste(gt_img, (panel_width, header_height))
    final_img.paste(overlay_img, (2 * panel_width, header_height))

    return final_img


def build_qwen3vl_prompt(action, object_category, is_thinking_model=False):
    """
    Build Qwen3VL grounding prompt for person-object pair detection.

    Args:
        action: Action verb (e.g., "riding", "sitting on")
        object_category: Object category (e.g., "bicycle", "bench")
        is_thinking_model: Whether to add thinking instructions (default: False)

    Returns:
        List of message dicts for Qwen3VL
    """
    # Base task and instructions
    prompt_text = (
        f"Task: Detect all person-{object_category} pairs where the person is {action} the {object_category}.\n\n"
        f"Instructions:\n"
        f"1. Identify ALL persons performing the action '{action}' with a {object_category}\n"
        f"2. For each person, identify the specific {object_category} they are interacting with\n"
        f"3. Return bounding boxes in [x1, y1, x2, y2] format\n\n"
    )

    # Add thinking section ONLY for thinking models
    # This instructs the model to put reasoning in <think> tags and output only JSON after </think>
    if is_thinking_model:
        prompt_text += (
            f"IMPORTANT - Use this exact format:\n"
            f"<think>\n"
            f"Step 1 - Analyze the image:\n"
            f"- Identify each person and {object_category} (visual cues: shape, color, pose, location)\n"
            f"- Determine which person is '{action}' which {object_category} (spatial relationship, interaction context)\n"
            f"- Decide precise bounding boxes for each person-{object_category} pair\n"
            f"</think>\n\n"
            f"Then output ONLY the JSON array (no other text).\n\n"
        )

    # Output format (same for both model types)
    prompt_text += (
        f"Output Format (JSON only, no other text):\n"
        f"[\n"
        f'  {{"pair_id": 1, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}},\n'
        f'  {{"pair_id": 2, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}}\n'
        f"]\n\n"
        f"Important:\n"
        f"- If NO pairs found, return: []\n"
        f"- Output ONLY the JSON array (no markdown, no explanations)\n"
        f"- Ensure coordinates are integers in [x1, y1, x2, y2] format"
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


def run_qwen3vl_inference(model, processor, image_path, action, object_category, is_thinking_model=False):
    """
    Run Qwen3VL inference for grounding task.

    Args:
        model: Qwen3VL model
        processor: Qwen3VL processor
        image_path: Path to image file
        action: Action verb
        object_category: Object category name
        is_thinking_model: Whether the model is a thinking model (default: False)

    Returns:
        thinking_content: Reasoning process (empty string for instruct models)
        output_text: Generated response text (final answer for thinking models)
        image: PIL Image object
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Build prompt
    messages = build_qwen3vl_prompt(action, object_category, is_thinking_model)

    # Replace placeholder with actual image
    for msg in messages:
        for content in msg["content"]:
            if content.get("type") == "image":
                content["image"] = image

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
    # Thinking models need more tokens (thinking content + JSON) and sampling enabled
    if is_thinking_model:
        max_tokens = 2048
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'do_sample': True,
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 20,
        }
    else:
        max_tokens = 512
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'do_sample': False,
            'temperature': None,
        }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs
        )

    # Extract response text (handles both thinking and instruct models)
    thinking_content, output_text = extract_response_from_output(
        generated_ids, inputs.input_ids, processor, is_thinking_model
    )

    return thinking_content, output_text, image


def eval_model(args):
    """Main evaluation function"""

    print("=" * 80)
    print("SWIG-HOI Grounding Evaluation (Qwen3VL)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Device:      {args.device}")
    print(f"Annotation:  {args.ann_file}")
    print(f"Images:      {args.img_prefix}")
    print(f"Output:      {args.result_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    print("=" * 80)
    print()

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load Qwen3VL model
    print(f"Loading Qwen3VL model: {args.model_name}")
    print(f"Target device: {args.device}")

    # Handle device mapping
    # When CUDA_VISIBLE_DEVICES is set, we need to use the local device index
    # e.g., if CUDA_VISIBLE_DEVICES=1, then cuda:0 refers to physical GPU 1
    if args.device == "auto":
        device_map = "auto"
    elif args.device.startswith("cuda"):
        # Extract device index
        if ":" in args.device:
            device_idx = args.device.split(":")[1]
        else:
            device_idx = "0"

        # Check if CUDA_VISIBLE_DEVICES is set
        import os
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

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(f"✓ Model loaded successfully on device: {model.device}")

    # Detect model type (thinking vs instruct)
    is_thinking_model = "Thinking" in args.model_name or "thinking" in args.model_name
    model_type = "Thinking" if is_thinking_model else "Instruct"
    print(f"✓ Model type: {model_type}")
    if is_thinking_model:
        print(f"  Note: Thinking model detected - will extract final answer after </think> token")
    print()

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")

        # Check if user is logged in
        try:
            # Try to login (will use cached credentials if available)
            wandb.login()

            # Initialize run
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"swig_ground_qwen3vl_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "dataset": "SWIG-HOI-Ground",
                    "task": "multi_pair_grounding",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["swig", "grounding", "qwen3vl", "multi-pair", "person-person"]
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
        print(f"✓ Visualization directory: {viz_dir}\n")

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        file_name = sample['file_name']
        action = sample['action']
        object_category = sample['object_category']
        img_path = os.path.join(args.img_prefix, file_name)
        img_shape = (sample['height'], sample['width'])

        # Build GT pairs from sample boxes
        # Each sample has 'boxes' which is a list of boxes: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        # Boxes are grouped in pairs: [person_box, object_box, person_box, object_box, ...]
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

        # Build prompt for logging
        prompt_messages = build_qwen3vl_prompt(action, object_category, is_thinking_model)
        prompt_text = prompt_messages[0]['content'][1]['text']  # Extract the text part

        # Run Qwen3VL inference
        thinking_content, output_text, image = run_qwen3vl_inference(
            model, processor, img_path, action, object_category, is_thinking_model
        )

        if show_verbose:
            print(f"  Prompt: {prompt_text[:100]}...")
            if thinking_content:
                print(f"  Thinking: {thinking_content[:150]}...")
            print(f"  Response: {output_text[:150]}...")

        # Parse predicted pairs
        pred_pairs = parse_qwen3vl_json_response(output_text, img_shape)

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
            'thinking_content': thinking_content,
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

        # Generate visualization for IoU=0.5 (moved outside IoU loop to ensure it runs once per sample)
        if viz_dir is not None:
            # Get matches at IoU=0.5 for visualization
            matches_05 = sample_result['matches_per_threshold'][0.5]
            matches_05_list, _, _ = match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5)

            try:
                viz_img = visualize_qwen3vl_grounding(
                    img_path, pred_pairs, gt_pairs, matches_05_list,
                    action, object_category, iou_threshold=0.5
                )

                # Save visualization with unique filename including action and object
                # This prevents overwriting when same image has multiple action-object pairs
                base_name = os.path.splitext(file_name)[0]
                # Sanitize action and object names for filename
                action_safe = action.replace(' ', '_').replace('/', '_')
                object_safe = object_category.replace(' ', '_').replace('/', '_')
                viz_filename = f"{base_name}_{action_safe}_{object_safe}_viz.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                viz_img.save(viz_path, quality=90)

                if show_verbose:
                    print(f"  Visualization saved: {viz_filename}")

                # Log to WandB if enabled
                if use_wandb:
                    log_dict = {
                        f"visualization/{idx:04d}_{action_safe}_{object_safe}": wandb.Image(
                            viz_img,
                            caption=f"{file_name} | {action} {object_category} | Pred:{len(pred_pairs)} GT:{len(gt_pairs)} Matched:{len(matches_05_list)}"
                        )
                    }

                    # Log thinking content if available
                    if thinking_content:
                        log_dict[f"thinking/{idx:04d}_{action_safe}_{object_safe}"] = wandb.Html(
                            f"<div style='font-family: monospace; white-space: pre-wrap; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
                            f"<h3>{file_name} - {action} {object_category}</h3>"
                            f"<h4>Thinking Process:</h4>"
                            f"{thinking_content}"
                            f"</div>"
                        )
                        log_dict[f"thinking_length/{idx:04d}"] = len(thinking_content.split())

                    wandb.log(log_dict)
            except Exception as e:
                if show_verbose:
                    print(f"  Warning: Visualization failed: {e}")

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
    print("SWIG-HOI Grounding Evaluation Results (Qwen3VL)")
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

    # Print detailed results
    print("\n" + "=" * 80)
    print("Average Recall (AR) Metrics")
    print("=" * 80)
    print(f"{'Metric':<12} {'Value':>10}  {'Description':<50}")
    print("-" * 80)
    print(f"{'AR':<12} {metrics['AR']*100:>9.1f}%  {'Average Recall @ IoU=0.50:0.95':<50}")
    print(f"{'AR@0.5':<12} {metrics['AR@0.5']*100:>9.1f}%  {'Average Recall @ IoU=0.50':<50}")
    print(f"{'AR@0.75':<12} {metrics['AR@0.75']*100:>9.1f}%  {'Average Recall @ IoU=0.75':<50}")
    print(f"{'ARs':<12} {metrics['ARs']*100:>9.1f}%  {'Average Recall for small objects (area < 32²)':<50}")
    print(f"{'ARm':<12} {metrics['ARm']*100:>9.1f}%  {'Average Recall for medium objects (32² < area < 96²)':<50}")
    print(f"{'ARl':<12} {metrics['ARl']*100:>9.1f}%  {'Average Recall for large objects (area > 96²)':<50}")
    print("-" * 80)

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        wandb.log({
            'total_samples': len(dataset_samples),
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

    # Save thinking content separately (only for samples with thinking)
    thinking_file = args.result_file.replace('.json', '_thinking.jsonl')
    thinking_samples = [s for s in per_sample_results if s.get('thinking_content')]
    if thinking_samples:
        print(f"Saving thinking content to: {thinking_file}")
        with open(thinking_file, 'w') as f:
            for result in thinking_samples:
                thinking_entry = {
                    'file_name': result['file_name'],
                    'action': result['action'],
                    'object': result['object'],
                    'thinking_content': result['thinking_content'],
                    'generated_text': result['generated_text']
                }
                f.write(json.dumps(thinking_entry) + '\n')
        print(f"  Total samples with thinking: {len(thinking_samples)}/{len(per_sample_results)}")

    # Save and log visualizations summary
    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith('.jpg')])
        print(f"\nVisualizations: {viz_dir}/")
        print(f"  Total images saved: {viz_count} (should match {len(dataset_samples)} samples)")
        if viz_count != len(dataset_samples):
            print(f"  ⚠️  WARNING: Expected {len(dataset_samples)} visualizations but got {viz_count}")

    if use_wandb:
        wandb.save(metrics_file)
        wandb.save(args.result_file)
        wandb.save(action_stats_file)
        if thinking_samples:
            wandb.save(thinking_file)

        # Create summary table for top/bottom performing actions
        action_table_data = []
        sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['total_samples'], reverse=True)
        for action, stats in sorted_actions[:20]:
            recall_05 = stats['matched_pairs_05'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
            action_table_data.append([
                action,
                stats['total_samples'],
                stats['total_gt_pairs'],
                stats['total_pred_pairs'],
                stats['matched_pairs_05'],
                f"{recall_05:.1%}"
            ])

        wandb.log({
            "action_performance_table": wandb.Table(
                columns=["Action", "Samples", "GT Pairs", "Pred Pairs", "Matched@0.5", "Recall@0.5"],
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
    parser = argparse.ArgumentParser(description="SWIG-HOI Grounding Evaluation with Qwen3VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3VL model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to SWIG grounding annotation file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Path to SWIG images directory")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-sample results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="swig-grounding-qwen3vl",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    eval_model(args)
