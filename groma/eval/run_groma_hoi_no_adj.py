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

Features:
- ✅ Checkpoint/Resume: Automatically saves progress and resumes from crashes
- ✅ Comprehensive Logging: Detailed log files with error tracking
- ✅ Error Recovery: Continues processing even if individual images fail
- ✅ Progress Tracking: Real-time progress bar and periodic checkpoint saves

Usage:
    # Single image evaluation
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --output-dir {output_directory} \
        --gpu 0

    # Dataset evaluation
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset {hico/swig} \
        --data-root {path_to_dataset} \
        --output-dir {output_directory} \
        --gpu 0

    # Resume from crash (finds checkpoint automatically)
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset hico \
        --data-root {path_to_dataset} \
        --resume-from {output_dir/YYYY-MM-DD_HH-MM-SS}

    # Custom checkpoint interval (save every 100 images)
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset hico \
        --data-root {path_to_dataset} \
        --output-dir {output_directory} \
        --checkpoint-interval 100

    # Single image with ground truth comparison
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --eval-dataset {hico/swig} \
        --data-root {path_to_dataset} \
        --output-dir {output_directory} \
        --gpu 0

    # Use multiple GPUs
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset {hico/swig} \
        --data-root {path_to_dataset} \
        --output-dir {output_directory} \
        --gpu 0,1

    # With Weights & Biases logging
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset hico \
        --data-root {path_to_dataset} \
        --output-dir {output_directory} \
        --gpu 0 \
        --use-wandb \
        --wandb-project groma-hoi \
        --wandb-entity {your_username} \
        --wandb-run-name hico_eval_v1 \
        --wandb-tags hico baseline

Output Structure:
    output_dir/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── checkpoint.pkl                    # Progress checkpoint for resume
        ├── evaluation.log                    # Detailed execution log
        ├── failed_images.json                # List of images that failed processing
        ├── {dataset}_evaluation_results.json # Final evaluation metrics and results
        ├── hoi_triplets/                     # Individual HOI visualizations
        └── comparison/                       # Ground truth vs prediction comparisons
"""

import os
import sys
import argparse

# Parse GPU argument FIRST, before any CUDA/PyTorch imports
# This is critical because CUDA_VISIBLE_DEVICES must be set before CUDA initializes
if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f"🔧 Setting CUDA_VISIBLE_DEVICES={gpu_id} (before imports)")
            break

# Now safe to import modules that use PyTorch/CUDA
from groma.eval.hoi_eval import create_orchestrator
from PIL import Image

# Try to import wandb (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. Install with: pip install wandb")


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

        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "single_image/precision": metrics['precision'],
                "single_image/recall": metrics['recall'],
                "single_image/true_positives": metrics['true_positives'],
                "single_image/total_predictions": metrics['total_predictions'],
                "single_image/total_gt": metrics['total_gt'],
                "single_image/num_triplets": len(result['triplets'])
            })

            # Log visualization if available
            import glob
            vis_files = glob.glob(os.path.join(orchestrator.timestamped_output_dir, "**/*.jpg"), recursive=True)
            if vis_files:
                for vis_file in vis_files[:5]:  # Log first 5 visualizations
                    wandb.log({
                        f"visualizations/{os.path.basename(vis_file)}": wandb.Image(vis_file)
                    })

    print(f"📁 Results saved to timestamped directory: {orchestrator.timestamped_output_dir}")
    return result


def eval_dataset(args):
    """Evaluate an entire dataset using the new modular system"""
    orchestrator = create_orchestrator(args)

    # Pass wandb logging function if enabled
    wandb_log_func = wandb.log if (args.use_wandb and WANDB_AVAILABLE) else None

    result = orchestrator.evaluate_dataset(
        args.dataset,
        args.data_root,
        getattr(args, 'max_images', None),
        getattr(args, 'batch_size', 1),
        wandb_log=wandb_log_func
    )

    print(f"\n🎉 Dataset evaluation completed!")
    print(f"📊 Final Results:")
    print(f"   Processed: {result['processed_count']}/{result['total_count']} images")

    if result['metrics']:
        print(f"   Overall Performance:")
        wandb_metrics = {}
        for metric, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"     {metric}: {value:.3f}")
                else:
                    print(f"     {metric}: {value}")

                # Prepare for wandb logging
                if args.use_wandb:
                    wandb_metrics[f"dataset/{metric}"] = value

        # Log final metrics to wandb
        if args.use_wandb and wandb_metrics:
            wandb.log(wandb_metrics)

            # Log summary metrics
            wandb.run.summary.update({
                "final_processed_count": result['processed_count'],
                "final_total_count": result['total_count'],
                **{k.replace('dataset/', ''): v for k, v in wandb_metrics.items()}
            })

            # Log sample visualizations
            import glob
            vis_files = glob.glob(os.path.join(orchestrator.timestamped_output_dir, "**/*.jpg"), recursive=True)
            if vis_files:
                sample_vis = vis_files[:10]  # Log 10 sample visualizations
                for i, vis_file in enumerate(sample_vis):
                    wandb.log({
                        f"sample_visualizations/image_{i}": wandb.Image(vis_file)
                    })

            # Save results JSON to wandb
            import glob
            json_files = glob.glob(os.path.join(orchestrator.timestamped_output_dir, "*.json"))
            for json_file in json_files:
                wandb.save(json_file)

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
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device ID to use (e.g., '0', '1', '0,1' for multiple GPUs). Default: use all available GPUs")

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

    # Checkpoint and resume support
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                       help="Save checkpoint every N images (default: 50)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume evaluation from a previous timestamped output directory")

    # Evaluation protocol settings
    parser.add_argument("--evaluation-mode", type=str, choices=['default', 'known_object'],
                       default='default',
                       help="Evaluation mode: 'default' (evaluate on all test images) or 'known_object' (evaluate only on images containing the target object)")

    # Weights & Biases logging
    parser.add_argument("--use-wandb", action='store_true',
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="groma-hoi-evaluation",
                       help="W&B project name (default: groma-hoi-evaluation)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (username or team name)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb-tags", type=str, nargs='+', default=None,
                       help="W&B tags for the run (e.g., --wandb-tags hico baseline)")

    args = parser.parse_args()

    # GPU device is already set at the top of the file before imports
    # Display which GPU is being used
    if args.gpu:
        print(f"🔧 Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    else:
        print(f"🔧 Using default GPUs (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all available')})")

    # Validate arguments
    if args.dataset and not args.data_root:
        parser.error("--data-root is required when using --dataset")

    if args.eval_dataset and not args.data_root:
        parser.error("--data-root is required when using --eval-dataset")

    # Handle resume functionality
    if args.resume_from:
        if not os.path.isdir(args.resume_from):
            parser.error(f"--resume-from directory does not exist: {args.resume_from}")

        checkpoint_file = os.path.join(args.resume_from, "checkpoint.pkl")
        if not os.path.exists(checkpoint_file):
            parser.error(f"No checkpoint found in resume directory: {args.resume_from}")

        print(f"🔄 Resuming evaluation from: {args.resume_from}")
        # Override output_dir to use the resume directory (without timestamp)
        # We'll need to adjust the orchestrator to not create a new timestamped dir
        args._resume_mode = True
        args._resume_dir = args.resume_from
    else:
        args._resume_mode = False
        args._resume_dir = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Weights & Biases if enabled
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("❌ Error: wandb is not installed. Install with: pip install wandb")
            print("   Continuing without wandb logging...")
            args.use_wandb = False
        else:
            import datetime
            run_name = args.wandb_run_name or f"{args.dataset if args.dataset else 'single_image'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            wandb_config = {
                "model_name": args.model_name,
                "quantization": args.quant_type,
                "evaluation_mode": args.evaluation_mode,
                "batch_size": getattr(args, 'batch_size', 1),
                "dataset": getattr(args, 'dataset', 'single_image'),
                "max_images": getattr(args, 'max_images', None),
                "gpu": args.gpu,
            }

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=wandb_config,
                tags=args.wandb_tags,
            )
            print(f"📊 Weights & Biases initialized: {wandb.run.get_url()}")

    # Display configuration
    print("🚀 HOI Evaluation for Groma (Refactored Modular Version)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Quantization: {args.quant_type}")
    print(f"Evaluation Mode: {args.evaluation_mode.upper()}")
    if args.evaluation_mode == 'known_object':
        print("  ℹ️  Known Object: Filters test set to only images containing the target object")
        print("  ℹ️  (e.g., for 'ride bicycle', only evaluate on images that have bicycles)")
    else:
        print("  ℹ️  Default: Evaluates on ALL test images (harder - includes background rejection)")
    print(f"Base output directory: {args.output_dir}")

    if args.resume_from:
        print(f"🔄 Resume Mode: Continuing from {args.resume_from}")
    else:
        print(f"📅 Each run creates a timestamped subfolder (YYYY-MM-DD_HH-MM-SS) to prevent overwriting")

    print(f"💾 Checkpoint interval: Every {args.checkpoint_interval} images")
    print(f"📝 Logging: evaluation.log (detailed) + console output")

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

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
        print("📊 Weights & Biases run completed and synced")


if __name__ == "__main__":
    main()