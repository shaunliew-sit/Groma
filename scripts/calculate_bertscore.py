#!/usr/bin/env python3
"""
Calculate BERTScore for Existing Prediction Results

This script computes BERTScore metrics from pre-existing prediction result files,
allowing you to evaluate models without rerunning the entire experiment.

BERTScore leverages pre-trained BERT embeddings to compute semantic similarity
between predictions and references, providing P(recision), R(ecall), and F1 scores.

Usage:
======
# Basic usage with default settings (GPU 0, roberta-large)
python scripts/calculate_bertscore.py \
    --input results-new/hico_action_qwen3vl/hico_action_qwen3vl_results_20251031_101049_per_triplet.json \
    --output results-new/hico_action_qwen3vl/hico_action_qwen3vl_results_20251031_101049_per_triplet_bertscore.json \
    --batch-size 128 \
    --gpu 6

# Custom output path and batch size
python scripts/calculate_bertscore.py \
    --input results-new/hico_action_qwen3vl/hico_action_qwen3vl_results_20251031_101049_per_triplet.json \
    --output results-new/hico_action_qwen3vl/hico_action_qwen3vl_results_20251031_101049_per_triplet_bertscore.json \
    --gpu 0 \
    --batch-size 128

Model Recommendations:
=====================
- roberta-large (default): Good balance of accuracy and speed, ~4-6GB GPU memory
- microsoft/deberta-xlarge-mnli: Best performance but slower, ~12-16GB GPU memory
- distilbert-base-uncased: Faster but less accurate, ~2GB GPU memory

For short text like action descriptions ("sitting on bench", "riding horse"),
roberta-large works very well and is the recommended choice.

Output:
=======
The script generates two output files:
1. *_bertscore.json: Full results with BERTScore added to each entry
2. *_bertscore_metrics.json: Summary statistics (mean P, R, F1)
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean prediction/reference text for more accurate BERTScore calculation.
    
    Handles:
    - Markdown bold: **text** → text
    - Markdown italic: *text* → text
    - Markdown headers: # Header → Header
    - Extra whitespace and newlines
    - Empty or None values
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if text is None:
        return ""
    
    text = str(text)
    
    # Remove markdown bold: **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Remove markdown italic: *text* → text (but not ** which is already handled)
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
    
    # Remove markdown headers: # Header → Header
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown code blocks: `code` → code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore from prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/calculate_bertscore.py \\
      --input results-new/hico_action_qwen3vl/results_per_triplet.json \\
      --gpu 0

  # Use different model and batch size
  python scripts/calculate_bertscore.py \\
      --input results-new/hico_action_qwen3vl/results_per_triplet.json \\
      --gpu 2 \\
      --model microsoft/deberta-xlarge-mnli \\
      --batch-size 32
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input JSON file with predictions (must have 'prediction' and 'ground_truth' fields)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output JSON file (default: input_bertscore.json)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",
        help="BERTScore model to use (default: roberta-large)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for BERTScore computation (default: 64)"
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        default=True,
        help="Use baseline rescaling for more interpretable scores (default: True)"
    )
    parser.add_argument(
        "--no-rescale",
        action="store_false",
        dest="rescale",
        help="Disable baseline rescaling"
    )
    parser.add_argument(
        "--prediction-key",
        type=str,
        default="prediction",
        help="Key name for prediction field in JSON (default: 'prediction')"
    )
    parser.add_argument(
        "--reference-key",
        type=str,
        default="ground_truth",
        help="Key name for reference/ground truth field in JSON (default: 'ground_truth')"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language for BERTScore baseline rescaling (default: 'en' for English)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean text before scoring (remove markdown, extra whitespace) (default: True)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Disable text cleaning"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Set GPU device BEFORE importing bert_score (critical for proper GPU selection)
    # =========================================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("\n" + "=" * 60)
    print("BERTScore Calculation for Prediction Results")
    print("=" * 60)
    print(f"Input file:    {args.input}")
    print(f"GPU device:    {args.gpu}")
    print(f"Model:         {args.model}")
    print(f"Language:      {args.lang}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Rescale:       {args.rescale}")
    print(f"Clean text:    {args.clean}")
    print("=" * 60 + "\n")
    
    # Import bert_score after setting GPU
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Error: bert_score package not found.")
        print("Please install it with: pip install bert_score")
        return 1
    
    # =========================================================================
    # Load results
    # =========================================================================
    print("1. Loading prediction results...")
    
    if not os.path.exists(args.input):
        print(f"   ✗ Error: Input file not found: {args.input}")
        return 1
    
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    print(f"   ✓ Loaded {len(results)} entries")
    
    # Validate that required fields exist
    if len(results) > 0:
        sample = results[0]
        if args.prediction_key not in sample:
            print(f"   ✗ Error: Prediction key '{args.prediction_key}' not found in results")
            print(f"   Available keys: {list(sample.keys())}")
            return 1
        if args.reference_key not in sample:
            print(f"   ✗ Error: Reference key '{args.reference_key}' not found in results")
            print(f"   Available keys: {list(sample.keys())}")
            return 1
    
    # =========================================================================
    # Extract predictions and references
    # =========================================================================
    print("\n2. Extracting predictions and references...")
    
    # Extract raw text
    raw_predictions = [str(r[args.prediction_key]) for r in results]
    raw_references = [str(r[args.reference_key]) for r in results]
    
    # Apply cleaning if enabled
    if args.clean:
        print("   Cleaning text (removing markdown, extra whitespace)...")
        predictions = [clean_text(p) for p in raw_predictions]
        references = [clean_text(r) for r in raw_references]
        
        # Count how many were modified
        pred_modified = sum(1 for p, rp in zip(predictions, raw_predictions) if p != rp)
        ref_modified = sum(1 for r, rr in zip(references, raw_references) if r != rr)
        print(f"   ✓ Cleaned {pred_modified} predictions, {ref_modified} references")
    else:
        predictions = raw_predictions
        references = raw_references
    
    # Show some examples
    print(f"   Sample predictions and references:")
    for i in range(min(3, len(predictions))):
        if args.clean and (predictions[i] != raw_predictions[i] or references[i] != raw_references[i]):
            print(f"   [{i}] Pred: '{predictions[i]}' (was: '{raw_predictions[i][:50]}...')")
            print(f"        Ref:  '{references[i]}'")
        else:
            print(f"   [{i}] Pred: '{predictions[i]}' | Ref: '{references[i]}'")
    
    # =========================================================================
    # Calculate BERTScore
    # =========================================================================
    print(f"\n3. Calculating BERTScore...")
    print(f"   Model: {args.model}")
    print(f"   This may take a few minutes for large datasets...\n")
    
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=args.model,
        lang=args.lang,
        batch_size=args.batch_size,
        rescale_with_baseline=args.rescale,
        verbose=True
    )
    
    # Convert tensors to lists
    P = P.tolist()
    R = R.tolist()
    F1 = F1.tolist()
    
    # =========================================================================
    # Add BERTScore to each result
    # =========================================================================
    print("\n4. Adding BERTScore to results...")
    
    for i, result in enumerate(results):
        result['bertscore_precision'] = P[i]
        result['bertscore_recall'] = R[i]
        result['bertscore_f1'] = F1[i]
        # Store cleaned versions if cleaning was applied
        if args.clean:
            result['prediction_cleaned'] = predictions[i]
            result['reference_cleaned'] = references[i]
    
    print(f"   ✓ Added BERTScore to {len(results)} entries")
    
    # =========================================================================
    # Calculate aggregate statistics
    # =========================================================================
    avg_precision = sum(P) / len(P)
    avg_recall = sum(R) / len(R)
    avg_f1 = sum(F1) / len(F1)
    
    # Create metrics summary
    metrics = {
        'model': args.model,
        'lang': args.lang,
        'rescale_with_baseline': args.rescale,
        'num_samples': len(results),
        'bertscore_precision_mean': avg_precision,
        'bertscore_recall_mean': avg_recall,
        'bertscore_f1_mean': avg_f1,
        'bertscore_precision_min': min(P),
        'bertscore_precision_max': max(P),
        'bertscore_recall_min': min(R),
        'bertscore_recall_max': max(R),
        'bertscore_f1_min': min(F1),
        'bertscore_f1_max': max(F1),
        'input_file': str(args.input),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BERTScore Results Summary")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"Rescale:            {args.rescale}")
    print(f"Number of samples:  {len(results)}")
    print("-" * 60)
    print(f"Average Precision:  {avg_precision:.4f}")
    print(f"Average Recall:     {avg_recall:.4f}")
    print(f"Average F1:         {avg_f1:.4f}")
    print("-" * 60)
    print(f"Precision range:    [{min(P):.4f}, {max(P):.4f}]")
    print(f"Recall range:       [{min(R):.4f}, {max(R):.4f}]")
    print(f"F1 range:           [{min(F1):.4f}, {max(F1):.4f}]")
    print("=" * 60)
    
    # =========================================================================
    # Determine output paths
    # =========================================================================
    if args.output:
        output_path = args.output
    else:
        base_path = args.input.rsplit('.json', 1)[0]
        output_path = f"{base_path}_bertscore.json"
    
    metrics_path = output_path.rsplit('.json', 1)[0] + '_metrics.json'
    
    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n5. Saving results...")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    print(f"   Saving detailed results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Detailed results saved")
    
    # Save metrics summary
    print(f"   Saving metrics summary to: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics summary saved")
    
    print("\n" + "=" * 60)
    print("✅ BERTScore Calculation Complete!")
    print("=" * 60)
    print(f"Results:  {output_path}")
    print(f"Metrics:  {metrics_path}")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

