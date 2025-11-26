"""
HICO-DET Action Referring Evaluation Script

Evaluates HOI action prediction using METEOR and CIDEr metrics.
Task: Given (person, object) bounding boxes, predict the connecting action.

Metrics: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L

Usage:
    python groma/eval/eval_hico_action_referring.py \
        --model-name groma-7b-finetune \
        --img-prefix /path/to/hico/images/test2015 \
        --ann-file groma_data/hico_action_referring_test.json \
        --pred-file results/predictions.json \
        --metrics-file results/metrics.json \
        --verbose \
        --max-images 10
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoImageProcessor

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.data.datasets.hico_action_referring import HICOActionReferringTest, collate_fn
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


def eval_model(args):
    """Main evaluation function"""

    print("=" * 80)
    print("HICO-DET Action Referring Evaluation (METEOR & CIDEr)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    print(f"Batch size:  {args.batch_size}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    if args.verbose:
        print(f"Verbose:     ENABLED (per-triplet results + visualizations)")
    print("=" * 80)
    print()

    # Initialize W&B if requested
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("⚠️  W&B requested but not installed. Install with: pip install wandb")
            print("    Continuing without W&B logging...\n")
            args.wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico-action-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": args.model_name,
                    "dataset": "HICO-DET",
                    "task": "action_referring",
                    "max_images": args.max_images,
                    "batch_size": args.batch_size,
                    "verbose": args.verbose
                }
            )
            print(f"✓ W&B logging enabled: {wandb.run.name}\n")

    # Initialize model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f"Loading model from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.eval()

    print(f"✓ Model loaded successfully\n")

    # Load dataset
    print(f"Loading dataset...")
    dataset = HICOActionReferringTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        test_mode=True,
        conv_temp='llava'
    )

    # Limit dataset if requested (for debugging)
    original_size = len(dataset)
    if args.max_images is not None and args.max_images < len(dataset):
        print(f"\n⚠️  DEBUG MODE: Limiting to first {args.max_images} triplets (out of {original_size})")
        dataset.data_infos = dataset.data_infos[:args.max_images]
        dataset.triplet_ids = dataset.triplet_ids[:args.max_images]

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=sampler,
        collate_fn=collate_fn
    )

    print(f"\nStarting inference on {len(dataset)} triplets...")
    print("=" * 80)
    print()

    # Storage for results
    predictions = []  # For COCO evaluation
    per_triplet_results = []  # For detailed analysis

    # Visualization setup
    viz_dir = None
    if args.verbose:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_dir = os.path.dirname(args.pred_file) if args.pred_file else "results"
        viz_dir = os.path.join(pred_dir, f"visualizations_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}\n")

    # Inference loop
    for batch in tqdm(dataloader, desc="Generating action predictions", unit="triplet"):
        input_ids = batch['input_ids'].cuda()
        image = batch['image'].cuda()
        triplet_id = batch['triplet_id']
        person_bbox = batch['person_bbox'].cuda()
        object_bbox = batch['object_bbox'].cuda()
        gt_action = batch['gt_action']
        img_info = batch['img_info']

        # Prepare refer_boxes: stack person and object bboxes
        # Shape: (2, 4) where first row is person, second is object
        refer_boxes = torch.stack([person_bbox, object_bbox], dim=0).unsqueeze(0)  # (1, 2, 4)

        # Model inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                refer_boxes=[refer_boxes.squeeze(0)],  # Pass as (2, 4)
                use_cache=True,
                do_sample=False,
                max_new_tokens=30,  # Actions are short phrases
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

        # Decode prediction
        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]

        # Get raw prediction (with special tokens for debugging)
        pred_raw = tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=False
        )[0]

        # Get clean prediction (remove special tokens)
        pred_clean = clean_action_response(pred_raw)

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
                "prediction_raw": pred_raw,
                "ground_truth": gt_action,
                "person_bbox": person_bbox.cpu().tolist(),
                "object_bbox": object_bbox.cpu().tolist(),
                "person_category": img_info.get('subject_category', 'person'),
                "object_category": img_info.get('object_category', 'object'),
                "image_file": img_info['file_name']
            })

            # Visualize triplet
            img_path = os.path.join(args.img_prefix, img_info['file_name'])
            viz_filename = f"{triplet_id.replace('/', '_')}_action.jpg"
            viz_path = os.path.join(viz_dir, viz_filename)

            visualize_action_triplet(
                img_path=img_path,
                person_bbox=person_bbox.cpu().tolist(),
                object_bbox=object_bbox.cpu().tolist(),
                pred_action=pred_clean,
                gt_action=gt_action,
                output_path=viz_path,
                person_category=img_info.get('subject_category', 'person'),
                object_category=img_info.get('object_category', 'object')
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

    coco_gt = COCO(args.ann_file)
    coco_pred = coco_gt.loadRes(args.pred_file)
    evaluator = COCOEvalCap(coco_gt, coco_pred)
    evaluator.params['image_id'] = coco_pred.getImgIds()
    evaluator.evaluate()

    # Print results
    print(f"\n{'='*80}")
    print("HICO-DET Action Referring Evaluation Results")
    print(f"{'='*80}")
    print(f"{'Metric':<15} {'Score':>10}  {'Raw':>10}")
    print("-" * 80)
    for metric, score in sorted(evaluator.eval.items()):
        percentage = score * 100
        print(f"{metric:<15} {percentage:>9.2f}%  {score:>10.4f}")
    print(f"{'='*80}\n")

    # Save metrics
    metrics = {
        "meteor": evaluator.eval['METEOR'] * 100,
        "cider": evaluator.eval['CIDEr'] * 100,
        "bleu_1": evaluator.eval.get('Bleu_1', 0.0) * 100,
        "bleu_2": evaluator.eval.get('Bleu_2', 0.0) * 100,
        "bleu_3": evaluator.eval.get('Bleu_3', 0.0) * 100,
        "bleu_4": evaluator.eval.get('Bleu_4', 0.0) * 100,
        "rouge_l": evaluator.eval.get('ROUGE_L', 0.0) * 100,
        "spice": evaluator.eval.get('SPICE', 0.0) * 100 if 'SPICE' in evaluator.eval else None,
        "num_triplets": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "model": args.model_name,
        "dataset": "HICO-DET Action Referring"
    }

    # Remove None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    metrics_file = args.metrics_file if args.metrics_file else args.pred_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}\n")

    # Log to W&B
    if args.wandb:
        wandb.log(metrics)
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

        # Regenerate visualizations with semantic similarity scores
        print(f"Regenerating visualizations with semantic similarity scores...")
        for result in tqdm(per_triplet_results, desc="Updating visualizations", unit="triplet"):
            triplet_id = result['triplet_id']

            # Get scores for this triplet
            if triplet_id in evaluator.imgToEval:
                scores = evaluator.imgToEval[triplet_id]
                meteor_score = scores['METEOR'] * 100  # Convert to percentage
                cider_score = scores['CIDEr'] * 100

                # Regenerate visualization with scores
                img_path = os.path.join(args.img_prefix, result['image_file'])
                viz_filename = f"{triplet_id.replace('/', '_')}_action.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)

                visualize_action_triplet(
                    img_path=img_path,
                    person_bbox=result['person_bbox'],
                    object_bbox=result['object_bbox'],
                    pred_action=result['prediction'],
                    gt_action=result['ground_truth'],
                    output_path=viz_path,
                    person_category=result['person_category'],
                    object_category=result['object_category'],
                    meteor_score=meteor_score,
                    cider_score=cider_score
                )
        print(f"✓ Visualizations updated with similarity scores\n")

        # Log per-action stats to W&B
        if args.wandb:
            # Log top/bottom 5 actions
            by_meteor = sorted(per_action_stats.items(), key=lambda x: x[1]['meteor'])
            wandb.log({
                "worst_action_meteor": by_meteor[0][1]['meteor'],
                "best_action_meteor": by_meteor[-1][1]['meteor'],
                "per_action_breakdown": wandb.Table(
                    columns=["action", "meteor", "cider", "count"],
                    data=[[action, stats['meteor'], stats['cider'], stats['count']]
                          for action, stats in sorted(per_action_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]]
                )
            })

    # Upload artifacts to W&B
    if args.wandb:
        print(f"\nUploading artifacts to W&B...")

        # Upload predictions JSON
        wandb.save(args.pred_file, base_path=os.path.dirname(args.pred_file) or '.')
        print(f"  ✓ Uploaded: {os.path.basename(args.pred_file)}")

        # Upload metrics JSON
        wandb.save(metrics_file, base_path=os.path.dirname(metrics_file) or '.')
        print(f"  ✓ Uploaded: {os.path.basename(metrics_file)}")

        # Upload log file (if provided)
        if hasattr(args, 'log_file') and args.log_file and os.path.exists(args.log_file):
            wandb.save(args.log_file, base_path=os.path.dirname(args.log_file) or '.')
            print(f"  ✓ Uploaded: {os.path.basename(args.log_file)}")

        # Upload per-triplet and per-action results (verbose mode)
        if args.verbose and len(per_triplet_results) > 0:
            wandb.save(per_triplet_file, base_path=os.path.dirname(per_triplet_file) or '.')
            print(f"  ✓ Uploaded: {os.path.basename(per_triplet_file)}")

            wandb.save(per_action_file, base_path=os.path.dirname(per_action_file) or '.')
            print(f"  ✓ Uploaded: {os.path.basename(per_action_file)}")

            # Upload visualizations as W&B Images
            print(f"  Uploading visualizations...")
            viz_images = []
            import glob
            viz_files = sorted(glob.glob(os.path.join(viz_dir, "*.jpg")))[:20]  # Upload first 20 visualizations

            for viz_file in viz_files:
                triplet_id = os.path.basename(viz_file).replace('_action.jpg', '')
                # Find corresponding result
                result = next((r for r in per_triplet_results if r['triplet_id'].replace('/', '_') == triplet_id), None)

                if result and triplet_id in evaluator.imgToEval:
                    scores = evaluator.imgToEval[result['triplet_id']]
                    caption = f"{result['image_file']} | Pred: {result['prediction']} | GT: {result['ground_truth']} | METEOR: {scores['METEOR']*100:.1f}%"
                else:
                    caption = os.path.basename(viz_file)

                viz_images.append(wandb.Image(viz_file, caption=caption))

            wandb.log({"visualizations": viz_images})
            print(f"  ✓ Uploaded {len(viz_images)} visualization images")

        # Create summary table with all results
        results_table = wandb.Table(
            columns=["Metric", "Score (%)"],
            data=[
                ["METEOR", f"{metrics['meteor']:.2f}"],
                ["CIDEr", f"{metrics['cider']:.2f}"],
                ["BLEU-1", f"{metrics['bleu_1']:.2f}"],
                ["BLEU-2", f"{metrics['bleu_2']:.2f}"],
                ["BLEU-3", f"{metrics['bleu_3']:.2f}"],
                ["BLEU-4", f"{metrics['bleu_4']:.2f}"],
                ["ROUGE-L", f"{metrics['rouge_l']:.2f}"],
            ]
        )
        wandb.log({"final_results": results_table})
        print(f"  ✓ Uploaded final results table")

        print(f"✓ All artifacts uploaded to W&B\n")

    # Final summary
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Results summary:")
    print(f"  Total triplets:  {metrics['num_triplets']}")
    print(f"  METEOR:          {metrics['meteor']:.2f}%")
    print(f"  CIDEr:           {metrics['cider']:.2f}%")
    print(f"  BLEU-1:          {metrics['bleu_1']:.2f}%")
    print(f"  ROUGE-L:         {metrics['rouge_l']:.2f}%")
    print()
    print(f"Outputs:")
    print(f"  Predictions:     {args.pred_file}")
    print(f"  Metrics:         {metrics_file}")
    if args.verbose:
        print(f"  Per-triplet:     {per_triplet_file}")
        print(f"  Per-action:      {per_action_file}")
        print(f"  Visualizations:  {viz_dir}/")
    print("=" * 80)

    # Finish W&B run
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HICO-DET Action Referring Evaluation with METEOR/CIDEr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python groma/eval/eval_hico_action_referring.py \\
        --model-name groma-7b-finetune \\
        --img-prefix /path/to/hico/images/test2015 \\
        --ann-file groma_data/hico_action_referring_test.json \\
        --pred-file results/predictions.json \\
        --metrics-file results/metrics.json

    # Quick test with verbose mode (10 images)
    python groma/eval/eval_hico_action_referring.py \\
        --model-name groma-7b-finetune \\
        --img-prefix /path/to/hico/images/test2015 \\
        --ann-file groma_data/hico_action_referring_test.json \\
        --pred-file results/test_predictions.json \\
        --metrics-file results/test_metrics.json \\
        --max-images 10 \\
        --verbose

    # With W&B logging
    python groma/eval/eval_hico_action_referring.py \\
        --model-name groma-7b-finetune \\
        --img-prefix /path/to/hico/images/test2015 \\
        --ann-file groma_data/hico_action_referring_test.json \\
        --pred-file results/predictions.json \\
        --metrics-file results/metrics.json \\
        --wandb \\
        --wandb-project hico-action-eval \\
        --wandb-run-name test-run-1
        """
    )

    # Required arguments
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path to Groma model checkpoint"
    )
    parser.add_argument(
        "--img-prefix",
        type=str,
        required=True,
        help="Path to HICO-DET images directory (e.g., images/test2015)"
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        required=True,
        help="Path to COCO-format action referring annotations"
    )
    parser.add_argument(
        "--pred-file",
        type=str,
        required=True,
        help="Path to save predictions (COCO results format)"
    )

    # Optional arguments
    parser.add_argument(
        "--gt-file",
        type=str,
        default=None,
        help="Path to save ground truth (optional, for inspection)"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to save metrics JSON (default: pred_file with _metrics.json suffix)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1, only 1 is supported for generation)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit evaluation to first N triplets (for debugging)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode (per-triplet results + visualizations)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (for W&B upload)"
    )

    # W&B arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hico-action-referring",
        help="W&B project name (default: hico-action-referring)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated timestamp)"
    )

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size != 1:
        print("⚠️  Warning: Only batch_size=1 is supported for generation. Setting batch_size=1.")
        args.batch_size = 1

    eval_model(args)
