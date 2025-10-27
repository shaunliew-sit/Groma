"""
Helper utilities for HOI action referring evaluation.

Provides:
- clean_action_response: Clean model output to extract action phrases
- visualize_action_triplet: Create visualization of person/object/action
- compute_per_action_metrics: Calculate per-action METEOR/CIDEr breakdown
"""

import re
import os
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import numpy as np


def clean_action_response(response):
    """
    Clean model response to extract clean action phrase.

    Handles:
    - Special tokens (</s>, <s>, <|endoftext|>)
    - HTML-like tags (<roi>, </roi>)
    - Extra whitespace
    - Lowercasing for consistency

    Examples:
        Input:  "sitting on</s>"          → Output: "sitting on"
        Input:  "racing</s>"              → Output: "racing"
        Input:  "hold<|endoftext|>"       → Output: "hold"
        Input:  "  Riding  a horse </s>"  → Output: "riding a horse"

    Args:
        response: Raw model output string

    Returns:
        Cleaned action phrase (lowercase, no special tokens)
    """
    # Remove common special tokens
    response = response.replace("</s>", "")
    response = response.replace("<s>", "")
    response = response.replace("<|endoftext|>", "")
    response = response.replace("<pad>", "")

    # Remove any HTML-like tags (e.g., <roi>, </roi>, <ground_box>)
    response = re.sub(r'<.*?>', '', response)

    # Clean whitespace and convert to lowercase
    response = response.strip().lower()

    # Remove multiple spaces
    response = re.sub(r'\s+', ' ', response)

    return response


def visualize_action_triplet(img_path, person_bbox, object_bbox, pred_action, gt_action, output_path,
                              person_category="person", object_category="object",
                              meteor_score=None, cider_score=None):
    """
    Visualize action triplet with person/object bboxes and predicted/GT action labels.

    Creates a side-by-side comparison showing:
    - Person bbox (red)
    - Object bbox (blue)
    - Predicted action
    - Ground truth action
    - Semantic similarity scores (METEOR, CIDEr) if provided
    - Similarity level based on METEOR score

    Args:
        img_path: Path to original image
        person_bbox: [cx, cy, w, h] in normalized [0,1] coordinates (CENTER format)
        object_bbox: [cx, cy, w, h] in normalized [0,1] coordinates (CENTER format)
        pred_action: Predicted action string
        gt_action: Ground truth action string
        output_path: Path to save visualization
        person_category: Category name for person (default: "person")
        object_category: Category name for object (default: "object")
        meteor_score: METEOR score (0-100%) for semantic similarity (optional)
        cider_score: CIDEr score for corpus consensus (optional)

    Returns:
        Path to saved visualization
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to load font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_large = ImageFont.load_default()

    # Convert normalized to pixel coordinates if needed
    def to_pixels(bbox):
        """Convert bbox to pixel coords. Handles both normalized [0,1] and absolute coords.

        Expects bbox in [cx, cy, w, h] format (center coordinates) and converts to
        [x, y, w, h] format (top-left coordinates) in pixels.
        """
        cx, cy, w, h = bbox

        # Check if bbox is normalized (all values <= 1.5 to handle edge cases)
        if all(val <= 1.5 for val in bbox):
            # Normalized coordinates [cx, cy, w, h] -> convert to [x, y, w, h] pixels
            w_px = int(w * width)
            h_px = int(h * height)
            cx_px = int(cx * width)
            cy_px = int(cy * height)

            # Convert center to top-left
            x_px = cx_px - w_px // 2
            y_px = cy_px - h_px // 2
        else:
            # Already in absolute pixels [cx, cy, w, h] -> convert to [x, y, w, h]
            w_px, h_px = int(w), int(h)
            x_px = int(cx - w_px // 2)
            y_px = int(cy - h_px // 2)

        return [x_px, y_px, w_px, h_px]

    person_px = to_pixels(person_bbox)
    object_px = to_pixels(object_bbox)

    # Draw person bbox (red)
    person_rect = [person_px[0], person_px[1],
                   person_px[0] + person_px[2], person_px[1] + person_px[3]]
    draw.rectangle(person_rect, outline="#FF0000", width=4)

    # Draw person label
    person_label = f"PERSON ({person_category})"
    person_label_y = max(5, person_px[1] - 22)
    draw.rectangle([person_px[0], person_label_y - 2,
                    person_px[0] + len(person_label) * 8, person_label_y + 18],
                   fill="#FF0000")
    draw.text((person_px[0] + 2, person_label_y), person_label, fill="white", font=font_small)

    # Draw object bbox (blue)
    object_rect = [object_px[0], object_px[1],
                   object_px[0] + object_px[2], object_px[1] + object_px[3]]
    draw.rectangle(object_rect, outline="#0000FF", width=4)

    # Draw object label
    object_label = f"OBJECT ({object_category})"
    object_label_y = max(5, object_px[1] - 22)
    draw.rectangle([object_px[0], object_label_y - 2,
                    object_px[0] + len(object_label) * 8, object_label_y + 18],
                   fill="#0000FF")
    draw.text((object_px[0] + 2, object_label_y), object_label, fill="white", font=font_small)

    # Draw action comparison box at bottom
    # Adjust box height based on whether we have scores
    box_height = 110 if meteor_score is not None else 80
    box_y = height - box_height
    draw.rectangle([0, box_y, width, height], fill="white", outline="black", width=2)

    # Check if actions match exactly
    exact_match = pred_action.lower().strip() == gt_action.lower().strip()

    # Draw action labels
    y_offset = box_y + 10
    draw.text((10, y_offset), f"Predicted: {pred_action}", fill="black", font=font)
    y_offset += 25
    draw.text((10, y_offset), f"Ground Truth: {gt_action}", fill="black", font=font)
    y_offset += 25

    # Show semantic similarity scores if provided
    if meteor_score is not None:
        # Determine similarity level and color based on METEOR score
        if exact_match:
            status_text = "✓ EXACT MATCH"
            status_color = "#00AA00"  # Green
        elif meteor_score >= 80:
            status_text = f"✓ VERY HIGH SIMILARITY ({meteor_score:.1f}%)"
            status_color = "#00AA00"  # Green
        elif meteor_score >= 60:
            status_text = f"✓ HIGH SIMILARITY ({meteor_score:.1f}%)"
            status_color = "#66AA00"  # Yellow-green
        elif meteor_score >= 40:
            status_text = f"~ MODERATE SIMILARITY ({meteor_score:.1f}%)"
            status_color = "#AAAA00"  # Yellow
        elif meteor_score >= 20:
            status_text = f"~ LOW SIMILARITY ({meteor_score:.1f}%)"
            status_color = "#FF6600"  # Orange
        else:
            status_text = f"✗ VERY LOW SIMILARITY ({meteor_score:.1f}%)"
            status_color = "#FF0000"  # Red

        draw.text((10, y_offset), status_text, fill=status_color, font=font_large)
        y_offset += 25

        # Show detailed scores
        if cider_score is not None:
            scores_text = f"METEOR: {meteor_score:.1f}%  |  CIDEr: {cider_score:.1f}"
        else:
            scores_text = f"METEOR: {meteor_score:.1f}%"
        draw.text((10, y_offset), scores_text, fill="#666666", font=font_small)
    else:
        # Fallback to simple match/mismatch if no scores provided
        match_status = "✓ EXACT MATCH" if exact_match else "✗ MISMATCH"
        match_color = "#00AA00" if exact_match else "#FF0000"
        draw.text((10, y_offset), match_status, fill=match_color, font=font_large)

    # Save visualization
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path)

    return output_path


def compute_per_action_metrics(per_triplet_results, evaluator):
    """
    Compute per-action METEOR/CIDEr scores and statistics.

    Args:
        per_triplet_results: List of dicts with:
            - triplet_id: Unique ID
            - ground_truth: GT action
            - prediction: Predicted action
        evaluator: COCOEvalCap instance with imgToEval populated

    Returns:
        Dict of {action_name: {meteor, cider, count, meteor_scores, cider_scores}}
        where meteor/cider are mean scores and *_scores are lists of individual scores
    """
    action_stats = defaultdict(lambda: {
        'meteor_scores': [],
        'cider_scores': [],
        'bleu_scores': [],
        'rouge_scores': [],
        'count': 0
    })

    # Group by action and collect scores
    for result in per_triplet_results:
        gt_action = result['ground_truth']
        triplet_id = result['triplet_id']

        # Get scores for this triplet from evaluator
        if triplet_id in evaluator.imgToEval:
            scores = evaluator.imgToEval[triplet_id]

            action_stats[gt_action]['meteor_scores'].append(scores['METEOR'])
            action_stats[gt_action]['cider_scores'].append(scores['CIDEr'])
            action_stats[gt_action]['bleu_scores'].append(scores.get('Bleu_1', 0.0))
            action_stats[gt_action]['rouge_scores'].append(scores.get('ROUGE_L', 0.0))
            action_stats[gt_action]['count'] += 1

    # Compute averages
    result = {}
    for action, stats in action_stats.items():
        if stats['count'] > 0:
            result[action] = {
                'meteor': np.mean(stats['meteor_scores']) * 100,
                'meteor_std': np.std(stats['meteor_scores']) * 100,
                'cider': np.mean(stats['cider_scores']) * 100,
                'cider_std': np.std(stats['cider_scores']) * 100,
                'bleu': np.mean(stats['bleu_scores']) * 100,
                'rouge': np.mean(stats['rouge_scores']) * 100,
                'count': stats['count'],
                'meteor_scores': stats['meteor_scores'],  # Keep raw scores for analysis
                'cider_scores': stats['cider_scores']
            }

    return result


def print_per_action_summary(per_action_metrics, top_k=10):
    """
    Print formatted summary of per-action metrics.

    Args:
        per_action_metrics: Output from compute_per_action_metrics()
        top_k: Number of actions to show in each category (default: 10)
    """
    print(f"\n{'='*80}")
    print("PER-ACTION METRICS SUMMARY")
    print(f"{'='*80}")

    # Sort by frequency
    by_frequency = sorted(per_action_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"\nTop-{top_k} Most Frequent Actions:")
    print(f"{'Action':<20} {'METEOR':<10} {'CIDEr':<10} {'Count':<8}")
    print("-" * 80)
    for action, stats in by_frequency[:top_k]:
        print(f"{action:<20} {stats['meteor']:>6.2f}%   {stats['cider']:>6.2f}%   {stats['count']:<8}")

    # Sort by METEOR (ascending - worst performing)
    by_meteor_asc = sorted(per_action_metrics.items(), key=lambda x: x[1]['meteor'])
    print(f"\nBottom-{top_k} Actions by METEOR Score (Hardest to Predict):")
    print(f"{'Action':<20} {'METEOR':<10} {'CIDEr':<10} {'Count':<8}")
    print("-" * 80)
    for action, stats in by_meteor_asc[:top_k]:
        print(f"{action:<20} {stats['meteor']:>6.2f}%   {stats['cider']:>6.2f}%   {stats['count']:<8}")

    # Sort by METEOR (descending - best performing)
    by_meteor_desc = sorted(per_action_metrics.items(), key=lambda x: x[1]['meteor'], reverse=True)
    print(f"\nTop-{top_k} Actions by METEOR Score (Easiest to Predict):")
    print(f"{'Action':<20} {'METEOR':<10} {'CIDEr':<10} {'Count':<8}")
    print("-" * 80)
    for action, stats in by_meteor_desc[:top_k]:
        print(f"{action:<20} {stats['meteor']:>6.2f}%   {stats['cider']:>6.2f}%   {stats['count']:<8}")

    print(f"{'='*80}\n")


def save_per_action_metrics_json(per_action_metrics, output_file):
    """
    Save per-action metrics to JSON file.

    Args:
        per_action_metrics: Output from compute_per_action_metrics()
        output_file: Path to save JSON

    Returns:
        Path to saved file
    """
    import json

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for action, stats in per_action_metrics.items():
        serializable_metrics[action] = {
            'meteor_mean': float(stats['meteor']),
            'meteor_std': float(stats['meteor_std']),
            'cider_mean': float(stats['cider']),
            'cider_std': float(stats['cider_std']),
            'bleu_mean': float(stats['bleu']),
            'rouge_mean': float(stats['rouge']),
            'count': int(stats['count'])
            # Note: omitting raw score lists to keep file size manageable
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    return output_file
