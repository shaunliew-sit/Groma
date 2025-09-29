#!/usr/bin/env python3
"""
HOI Evaluation Script for Groma with HICO-DET and SWIG-HOI Support

This script processes images/datasets to extract HOI triplets and evaluate them
using the HICO-DET and SWIG-HOI evaluation protocols with proper format conversion.

This is a refactored version that uses modular components for better code organization
and reusability. The original implementation has been broken down into specialized modules:

- hoi_extractor.py: HOI triplet extraction using NLP analysis
- visualization.py: Visual representation of HOI predictions
- dataset_utils.py: Dataset loading and management utilities
- evaluation_utils.py: HOI detection metric calculations
- image_processing.py: Image processing and model interface
- evaluation_orchestrator.py: Main evaluation coordination logic

Usage:
    # Single image evaluation
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --output-dir {output_directory}

    # Dataset evaluation
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset {hico/swig} \
        --data-root {path_to_dataset} \
        --output-dir {output_directory}

    # Single image with ground truth comparison
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --eval-dataset {hico/swig} \
        --data-root {path_to_dataset} \
        --output-dir {output_directory}
"""

import os
import argparse

# Import the new modular evaluation system
from groma.eval.hoi_eval import create_orchestrator


def eval_single_image(args):
    """Evaluate a single image using the new modular system"""
    orchestrator = create_orchestrator(args)

    # Determine dataset context for evaluation
    dataset_type = getattr(args, 'eval_dataset', None)
    data_root = getattr(args, 'data_root', None)

    result = orchestrator.evaluate_single_image(
        args.image_file, dataset_type, data_root
    )

    print(f"\n🎉 Single image evaluation completed!")
    print(f"📝 Extracted {len(result['triplets'])} HOI triplets")

    if result['metrics']:
        metrics = result['metrics']
        print(f"📊 Evaluation Metrics:")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   True Positives: {metrics['true_positives']}")
        print(f"   Total Predictions: {metrics['total_predictions']}")
        print(f"   Total Ground Truth: {metrics['total_gt']}")

    print(f"📁 Results saved to timestamped directory: {orchestrator.timestamped_output_dir}")
    return result


def eval_dataset(args):
    """Evaluate an entire dataset using the new modular system"""
    orchestrator = create_orchestrator(args)

    result = orchestrator.evaluate_dataset(
        args.dataset,
        args.data_root,
        getattr(args, 'max_images', None),
        getattr(args, 'batch_size', 1)
    )

    print(f"\n🎉 Dataset evaluation completed!")
    print(f"📊 Final Results:")
    print(f"   Processed: {result['processed_count']}/{result['total_count']} images")

    if result['metrics']:
        print(f"   Overall Performance:")
        for metric, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"     {metric}: {value:.3f}")
                else:
                    print(f"     {metric}: {value}")

    print(f"📁 Detailed results saved to timestamped directory: {orchestrator.timestamped_output_dir}")
    return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HOI Evaluation for Groma with HICO-DET and SWIG-HOI Support")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/",
                       help="Path to Groma model")
    parser.add_argument("--quant_type", type=str, default='none',
                       choices=['none', 'fp16', '8bit', '4bit'],
                       help="Quantization type")

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-file", type=str,
                           help="Path to input image (single image mode)")
    input_group.add_argument("--dataset", type=str, choices=['hico', 'swig'],
                           help="Dataset to evaluate (dataset mode)")

    # Dataset arguments for evaluation context (optional for single image, required for dataset mode)
    parser.add_argument("--data-root", type=str,
                       help="Root directory of dataset (required for dataset mode, optional for single image with evaluation)")
    parser.add_argument("--eval-dataset", type=str, choices=['hico', 'swig'],
                       help="Dataset type for single image evaluation context (enables ground truth comparison)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default='hoi_evaluation_output',
                       help="Output directory for results")
    parser.add_argument("--save-pkl", action='store_true',
                       help="Save pickle files (legacy format). By default, only JSON results are saved.")

    # Evaluation control
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (default: None for full dataset)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing images (default: 8)")

    # Evaluation protocol settings
    parser.add_argument("--evaluation-mode", type=str, choices=['default', 'known_objects'],
                       default='default',
                       help="Evaluation mode: 'default' (predict all boxes + interactions) or 'known_objects' (use GT object boxes)")

    args = parser.parse_args()

    # Validate arguments
    if args.dataset and not args.data_root:
        parser.error("--data-root is required when using --dataset")

    if args.eval_dataset and not args.data_root:
        parser.error("--data-root is required when using --eval-dataset")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Display configuration
    print("🚀 HOI Evaluation for Groma (Refactored Modular Version)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Quantization: {args.quant_type}")
    print(f"Base output directory: {args.output_dir}")
    print(f"📅 Each run creates a timestamped subfolder (YYYY-MM-DD_HH-MM-SS) to prevent overwriting")

    # Run appropriate evaluation mode
    if args.image_file:
        if args.eval_dataset:
            print(f"Mode: Single image evaluation with {args.eval_dataset.upper()} ground truth comparison")
        else:
            print("Mode: Single image evaluation (no ground truth)")
        print(f"Image: {args.image_file}")
        eval_single_image(args)
    else:
        print(f"Mode: Dataset evaluation ({args.dataset.upper()})")
        print(f"Dataset root: {args.data_root}")
        if args.max_images:
            print(f"Max images: {args.max_images}")
        eval_dataset(args)


if __name__ == "__main__":
    main()