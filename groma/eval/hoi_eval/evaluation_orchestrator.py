"""Main evaluation orchestrator for HOI detection evaluation."""

import os
import json
import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel

from .hoi_extractor import POSBasedHOIExtractorNoAdj
from .visualization import HOIVisualizer
from .dataset_utils import load_dataset, find_image_in_dataset, extract_ground_truth_hois
from .evaluation_utils import calculate_single_image_metrics
from .image_processing import load_image, generate_hoi_response, generate_hoi_response_batch
from .hico_evaluator import HICOEvaluator
from .swig_evaluator import SWiGEvaluator


class HOIEvaluationOrchestrator:
    """Main orchestrator for HOI evaluation tasks."""

    def __init__(self, args):
        """Initialize the orchestrator with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.vis_processor = None
        self.hoi_extractor = None
        self.evaluator = None

        # Create timestamped output directory for this run
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamped_output_dir = None
        if hasattr(args, 'output_dir') and args.output_dir:
            # Create timestamped subfolder: output_dir/YYYY-MM-DD_HH-MM-SS/
            self.timestamped_output_dir = os.path.join(args.output_dir, self.timestamp)
            os.makedirs(self.timestamped_output_dir, exist_ok=True)
            print(f"📁 Created timestamped output directory: {self.timestamped_output_dir}")

            # Create organized subfolders within the timestamped directory
            self.hoi_triplets_dir = os.path.join(self.timestamped_output_dir, "hoi_triplets")
            self.comparison_dir = os.path.join(self.timestamped_output_dir, "comparison")
            os.makedirs(self.hoi_triplets_dir, exist_ok=True)
            os.makedirs(self.comparison_dir, exist_ok=True)

    def setup_model(self):
        """Setup the Groma model and related components."""
        print("🔧 Setting up model and processors...")

        disable_torch_init()
        model_name = os.path.expanduser(self.args.model_name)
        self.vis_processor = AutoImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        kwargs = {}
        if self.args.quant_type == 'fp16':
            kwargs['torch_dtype'] = torch.float16
        elif self.args.quant_type == '8bit':
            kwargs['load_in_8bit'] = True
        elif self.args.quant_type == '4bit':
            int4_quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4'
            )
            kwargs = {'quantization_config': int4_quant_cfg}

        if self.args.quant_type == '8bit' or self.args.quant_type == '4bit':
            self.model = GromaModel.from_pretrained(model_name, **kwargs)
        else:
            self.model = GromaModel.from_pretrained(model_name, **kwargs).cuda()
        self.model.init_special_token_id(self.tokenizer)

        # Initialize HOI extractor
        self.hoi_extractor = POSBasedHOIExtractorNoAdj()

        print("✅ Model setup completed")

    def setup_evaluator(self, dataset_type):
        """Setup the appropriate evaluator for the dataset."""
        # Use timestamped output directory if available, otherwise fallback to original
        output_dir = self.timestamped_output_dir or getattr(self.args, 'output_dir', 'hoi_evaluation_output')

        if dataset_type == 'hico':
            # HICO evaluator expects: anno_file, output_dir, evaluation_mode
            anno_file = os.path.join(self.args.data_root, 'annotations', 'test_hico_ann.json')
            evaluation_mode = getattr(self.args, 'evaluation_mode', 'default')
            self.evaluator = HICOEvaluator(anno_file, output_dir, evaluation_mode=evaluation_mode)
        elif dataset_type == 'swig':
            # SWIG evaluator expects: anno_file, output_dir
            anno_file = os.path.join(self.args.data_root, 'annotations', 'swig_test_1000.json')
            self.evaluator = SWiGEvaluator(anno_file, output_dir)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def evaluate_single_image(self, image_file, dataset_type=None, data_root=None):
        """Evaluate a single image for HOI detection."""
        print(f"\n🎯 SINGLE IMAGE EVALUATION")
        print(f"Image: {image_file}")

        # Load ground truth (should always be available for dataset evaluation)
        gt_data = None
        gt_hois = []
        if dataset_type and data_root:
            print(f"\n🔍 Loading ground truth from {dataset_type.upper()} dataset...")
            gt_data = find_image_in_dataset(image_file, dataset_type, data_root)
            if gt_data:
                gt_hois = extract_ground_truth_hois(gt_data, dataset_type)
                print(f"✅ Found ground truth with {len(gt_hois)} HOI annotations")
                if len(gt_hois) == 0:
                    print("⚠️ Warning: Ground truth loaded but contains 0 HOI annotations")
            else:
                print("❌ ERROR: No ground truth found for this image - check dataset annotations")
        else:
            print("\n📋 Running single image evaluation WITHOUT ground truth comparison")

        # Load and process image
        raw_image = load_image(image_file)
        image_width, image_height = raw_image.size

        # Resize to 448x448 (square) as required by Groma model
        processed_image = raw_image.resize((448, 448))

        # Process with vision encoder
        image_processed = self.vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_processed.unsqueeze(0).cuda()

        # Generate HOI response (using original prompt)
        hoi_query = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

        response_text, coordinates_info = generate_hoi_response(
            self.model, self.tokenizer, self.vis_processor, image_tensor, hoi_query
        )

        print(f"\n🤖 Model Response: {response_text}")
        print(f"🎯 Found {len(coordinates_info)} detected regions")

        # Extract HOI triplets
        entities = self.hoi_extractor.parse_grounded_response(response_text)

        # Fallback: if no entities found from response text but we have coordinates,
        # assume the model detected regions but didn't provide descriptions
        # In this case, we need to infer entity types from context
        if not entities and coordinates_info:
            print("DEBUG: No entities parsed from response text, trying coordinate-based inference")
            # For now, skip processing as the HOI extraction needs proper entity descriptions
            # The issue is that the model response format may have changed
            print("DEBUG: Cannot extract HOI triplets without proper entity descriptions")
            print(f"DEBUG: Model response was: '{response_text}'")
            print("DEBUG: Expected format: '<p>entity_description</p> <roi><r#></roi>'")
            print("DEBUG: This suggests the model prompt or response format has changed")

        triplets = self.hoi_extractor.extract_hoi_triplets(
            response_text, entities, coordinates_info, dataset_type or 'hico'
        )

        print(f"🔗 Extracted {len(triplets)} HOI triplets")

        # Debug triplet details
        triplets_with_bbox = sum(1 for t in triplets if t.get('human_bbox') and t.get('object_bbox'))
        print(f"🔲 Triplets with valid bounding boxes: {triplets_with_bbox}/{len(triplets)}")
        if len(triplets) > 0 and triplets_with_bbox == 0:
            print("⚠️ WARNING: No triplets have bounding boxes - predictions will be empty")
            print(f"🔍 Sample triplet keys: {list(triplets[0].keys()) if triplets else 'None'}")
            if coordinates_info:
                print(f"📍 Available coordinates: {len(coordinates_info)} regions")
            else:
                print("📍 No coordinate information available")

        # Prepare visualization triplets
        viz_triplets = self.hoi_extractor.prepare_visualization_triplets(
            triplets, dataset_type or 'hico'
        )

        # Convert to predictions for evaluation
        predictions = []
        if dataset_type:
            predictions = self.hoi_extractor.convert_triplets_to_predictions(
                triplets, gt_data['image_id'] if gt_data else 0,
                image_width, image_height, dataset_type
            )
            print(f"🔄 Converted {len(triplets)} triplets to {len(predictions)} predictions")
            if len(triplets) > 0 and len(predictions) == 0:
                print("⚠️ WARNING: Triplets found but no predictions generated - check conversion logic")
            elif len(predictions) > 0:
                print(f"✅ Sample prediction format: {predictions[0] if predictions else 'None'}")

        # Calculate metrics if ground truth available
        metrics = None
        if gt_hois and predictions:
            metrics = calculate_single_image_metrics(
                predictions, gt_hois, image_width, image_height
            )

        # Create visualizations
        if self.timestamped_output_dir:
            visualizer = HOIVisualizer(raw_image)

            # Use pre-created timestamped subfolders
            # Basic triplet visualization
            viz_path = os.path.join(self.hoi_triplets_dir,
                                  f"{os.path.splitext(os.path.basename(image_file))[0]}_hoi_triplets.jpg")
            visualizer.visualize_triplets(viz_triplets, coordinates_info, viz_path)

            # Comparison visualization (always create if GT data exists, even if 0 HOI annotations)
            if gt_data and dataset_type:
                comp_path = os.path.join(self.comparison_dir,
                                       f"{os.path.splitext(os.path.basename(image_file))[0]}_comparison.jpg")
                visualizer.visualize_comparison(
                    viz_triplets, coordinates_info, gt_hois, metrics, comp_path, dataset_type
                )

        return {
            'response_text': response_text,
            'coordinates_info': coordinates_info,
            'triplets': viz_triplets,
            'predictions': predictions,
            'ground_truth': gt_hois,
            'metrics': metrics
        }

    def evaluate_dataset(self, dataset_type, data_root, max_images=None, batch_size=1, wandb_log=None):
        """Evaluate an entire dataset with batch processing.

        Args:
            wandb_log: Optional wandb logging function to call with intermediate metrics
        """
        print(f"\n{'='*80}")
        print(f"DATASET EVALUATION: {dataset_type.upper()}")
        print(f"{'='*80}")

        # Load dataset
        dataset = load_dataset(dataset_type, data_root, max_images)
        if max_images:
            print(f"Loaded {len(dataset)} images from {dataset_type} dataset (limited to {max_images} max)")
        else:
            print(f"Loaded {len(dataset)} images from {dataset_type} dataset (full set)")

        # Setup evaluator
        self.setup_evaluator(dataset_type)

        # Memory management and batch size adjustment
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            free_memory = total_memory - allocated_memory

            print(f"GPU Memory info: Total: {total_memory:.2f} GB, Allocated: {allocated_memory:.2f} GB, Free: {free_memory:.2f} GB")

            # Adjust batch size based on available memory
            if free_memory < 4.0 and batch_size > 4:
                batch_size = 4
                print(f"⚠️  Reducing batch size to {batch_size} due to limited GPU memory")
            elif free_memory < 2.0 and batch_size > 2:
                batch_size = 2
                print(f"⚠️  Reducing batch size to {batch_size} due to very limited GPU memory")
            elif free_memory < 1.0:
                batch_size = 1
                print(f"⚠️  Reducing batch size to {batch_size} due to extremely limited GPU memory")

            torch.cuda.empty_cache()

        print(f"Starting evaluation on {len(dataset)} images with batch size {batch_size}...")

        all_predictions = {}
        detailed_results = {}
        processed_count = 0

        # Running metrics for wandb
        running_precision = []
        running_recall = []
        running_triplets = []

        # HOI query prompt
        hoi_query = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

        # Process dataset in batches
        from tqdm import tqdm
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=len(dataset), desc=f"Processing {dataset_type.upper()} images")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_samples = dataset[batch_start:batch_end]

            try:
                print(f"\n🔄 Processing batch {batch_idx + 1}/{num_batches} (images {batch_start + 1}-{batch_end})")

                # Load and preprocess images in batch
                batch_images = []
                batch_image_info = []

                for data_item in batch_samples:
                    image_path = data_item['image_path']
                    raw_image = load_image(image_path)

                    # Resize to 448x448 (square) as required by Groma model
                    processed_image = raw_image.resize((448, 448))
                    image_processed = self.vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'][0]
                    image_tensor = image_processed.unsqueeze(0)

                    batch_images.append(image_tensor)
                    batch_image_info.append({
                        'data_item': data_item,
                        'raw_image': raw_image,
                        'image_width': raw_image.size[0],
                        'image_height': raw_image.size[1]
                    })

                # Move batch to GPU
                batch_images = [img.cuda() for img in batch_images]

                # Generate responses for the batch
                batch_results = generate_hoi_response_batch(
                    self.model, self.tokenizer, self.vis_processor, batch_images, hoi_query, batch_size=len(batch_images)
                )

                # Process each result in the batch
                for i, (response_text, coordinates_info) in enumerate(batch_results):
                    data_item = batch_image_info[i]['data_item']
                    raw_image = batch_image_info[i]['raw_image']
                    image_width = batch_image_info[i]['image_width']
                    image_height = batch_image_info[i]['image_height']
                    image_id = data_item['image_id']
                    image_path = data_item['image_path']

                    print(f"  [{batch_start + i + 1}/{len(dataset)}] Processing image {image_id}: {os.path.basename(image_path)}")

                    # Load ground truth for this image
                    gt_data = data_item  # Dataset already includes ground truth
                    gt_hois = extract_ground_truth_hois(gt_data, dataset_type)
                    print(f"DEBUG: Orchestrator loaded {len(gt_hois)} ground truth HOIs for image {image_id}")

                    # Extract HOI triplets
                    entities = self.hoi_extractor.parse_grounded_response(response_text)
                    triplets = self.hoi_extractor.extract_hoi_triplets(
                        response_text, entities, coordinates_info, dataset_type
                    )

                    # Prepare visualization triplets
                    viz_triplets = self.hoi_extractor.prepare_visualization_triplets(triplets, dataset_type)

                    # Convert to predictions for evaluation
                    predictions = self.hoi_extractor.convert_triplets_to_predictions(
                        triplets, image_id, image_width, image_height, dataset_type
                    )

                    # Calculate metrics
                    metrics = None
                    print(f"DEBUG: Before metrics calculation - gt_hois: {len(gt_hois)}, predictions: {len(predictions)}")
                    if gt_hois:
                        print(f"DEBUG: Ground truth HOI IDs: {[gt['hoi_id'] for gt in gt_hois]}")
                    if predictions:
                        print(f"DEBUG: Prediction HOI IDs: {[pred[0] for pred in predictions]}")
                    if gt_hois and predictions:
                        print(f"DEBUG: Calling calculate_single_image_metrics with {len(gt_hois)} gt_hois and {len(predictions)} predictions")
                        metrics = calculate_single_image_metrics(predictions, gt_hois, image_width, image_height)
                    else:
                        print(f"DEBUG: Skipping metrics calculation - gt_hois: {len(gt_hois)}, predictions: {len(predictions)}")

                    # Store predictions for final evaluation
                    if predictions:
                        all_predictions[image_id] = predictions

                    # Create visualizations
                    if self.timestamped_output_dir:
                        visualizer = HOIVisualizer(raw_image)

                        # Basic triplet visualization
                        viz_path = os.path.join(self.hoi_triplets_dir,
                                             f"{os.path.splitext(os.path.basename(image_path))[0]}_hoi_triplets.jpg")
                        visualizer.visualize_triplets(viz_triplets, coordinates_info, viz_path)

                        # Comparison visualization
                        comp_path = os.path.join(self.comparison_dir,
                                               f"{os.path.splitext(os.path.basename(image_path))[0]}_comparison.jpg")
                        visualizer.visualize_comparison(
                            viz_triplets, coordinates_info, gt_hois, metrics, comp_path, dataset_type
                        )

                    # Store detailed results
                    detailed_results[image_id] = {
                        'file_name': os.path.basename(image_path),
                        'triplets_found': len(viz_triplets),
                        'predictions_count': len(predictions),
                        'gt_count': len(gt_hois),
                        'metrics': metrics,
                        'extracted_hoi_triplets': [
                            {
                                'human': triplet['human']['text'],
                                'human_region_id': triplet['human']['region_id'],
                                'action': triplet['action'],
                                'original_action': triplet.get('original_action', triplet['action']),
                                'object': triplet['object']['text'],
                                'object_region_id': triplet['object']['region_id'],
                                'confidence': triplet.get('confidence', 0.0),
                                'mapping_status': triplet.get('mapping_status', 'unknown'),
                                'hoi_id': triplet.get('hoi_id'),
                                'evaluation_eligible': triplet.get('evaluation_eligible', False)
                            }
                            for triplet in viz_triplets
                        ]
                    }

                    processed_count += 1
                    progress_bar.update(1)

                    # Track running metrics
                    if metrics:
                        running_precision.append(metrics.get('precision', 0))
                        running_recall.append(metrics.get('recall', 0))
                    running_triplets.append(len(viz_triplets))

                    # Log intermediate metrics to wandb
                    if wandb_log and metrics:
                        avg_precision = sum(running_precision) / len(running_precision) if running_precision else 0
                        avg_recall = sum(running_recall) / len(running_recall) if running_recall else 0
                        avg_triplets = sum(running_triplets) / len(running_triplets) if running_triplets else 0

                        wandb_log({
                            "progress/processed_images": processed_count,
                            "progress/current_precision": metrics.get('precision', 0),
                            "progress/current_recall": metrics.get('recall', 0),
                            "progress/current_triplets": len(viz_triplets),
                            "progress/avg_precision": avg_precision,
                            "progress/avg_recall": avg_recall,
                            "progress/avg_triplets": avg_triplets,
                        })

                    # Log sample visualizations periodically (every 100 images)
                    if wandb_log and processed_count % 100 == 0:
                        recent_vis = os.path.join(self.comparison_dir,
                                                 f"{os.path.splitext(os.path.basename(image_path))[0]}_comparison.jpg")
                        if os.path.exists(recent_vis):
                            try:
                                import wandb
                                wandb_log({
                                    f"progress_visualizations/sample_{processed_count}": wandb.Image(recent_vis)
                                })
                            except:
                                pass  # Skip if wandb not available

                # Log batch completion
                if wandb_log:
                    wandb_log({
                        "progress/batch_completed": batch_idx + 1,
                        "progress/total_batches": num_batches,
                        "progress/batch_progress": (batch_idx + 1) / num_batches * 100
                    })

                # Memory cleanup after batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error processing batch {batch_idx + 1}: {str(e)}")
                # Skip failed images in batch and continue with next batch
                for j in range(len(batch_samples)):
                    progress_bar.update(1)
                continue

        progress_bar.close()

        # Display processing summary (matching original script format)
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETED")
        print(f"{'='*50}")
        print(f"Total images in dataset: {len(dataset)}")
        print(f"Successfully processed: {len(all_predictions)}")
        print(f"Failed/skipped: {len(dataset) - len(all_predictions)}")
        print(f"Success rate: {len(all_predictions)/len(dataset)*100:.1f}%")
        print(f"{'='*50}")

        # Final evaluation using official evaluator
        print(f"\n📊 Running final evaluation on {processed_count} processed images...")
        print(f"🔢 Images with valid predictions: {len(all_predictions)}")

        # Debug prediction statistics
        if all_predictions:
            total_preds = sum(len(preds) for preds in all_predictions.values())
            avg_preds = total_preds / len(all_predictions) if all_predictions else 0
            print(f"📈 Total predictions across all images: {total_preds}")
            print(f"📊 Average predictions per image: {avg_preds:.1f}")

            # Show sample predictions
            sample_img_id = list(all_predictions.keys())[0]
            sample_preds = all_predictions[sample_img_id][:3]  # First 3 predictions
            print(f"🔍 Sample predictions from image {sample_img_id}:")
            for i, pred in enumerate(sample_preds):
                print(f"   [{i}] {pred}")

        # Update evaluator with all predictions (original script approach)
        if all_predictions:
            print("Updating evaluator...")
            self.evaluator.update(all_predictions)
            print("Computing metrics...")
            self.evaluator.accumulate()

            # Calculate console metrics manually BEFORE summarize() for consistency with printed output
            # This matches the backup code approach and ensures consistency between printed and JSON metrics
            console_metrics = {}
            if hasattr(self.evaluator, 'swig_ap'):  # SWIG dataset
                from .swig_v1_categories import SWIG_INTERACTIONS
                eval_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
                zero_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1 and x["frequency"] == 0])
                rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1 and x["evaluation"] == 1])
                nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2 and x["evaluation"] == 1])

                # Calculate metrics with same precision as displayed
                zero_shot_mAP = np.mean(self.evaluator.swig_ap[zero_hois])
                rare_mAP = np.mean(self.evaluator.swig_ap[rare_hois])
                nonrare_mAP = np.mean(self.evaluator.swig_ap[nonrare_hois])
                full_mAP = np.mean(self.evaluator.swig_ap[eval_hois])

                console_metrics = {
                    "zero_shot_mAP": float(zero_shot_mAP),
                    "rare_mAP": float(rare_mAP),
                    "nonrare_mAP": float(nonrare_mAP),
                    "full_mAP": float(full_mAP),
                    "zero_shot_mAP_percent": float(zero_shot_mAP * 100),
                    "rare_mAP_percent": float(rare_mAP * 100),
                    "nonrare_mAP_percent": float(nonrare_mAP * 100),
                    "full_mAP_percent": float(full_mAP * 100),
                    "dataset_type": "SWIG-HOI"
                }
            elif hasattr(self.evaluator, 'hico_ap'):  # HICO dataset
                # Call summarize() first - this will calculate and set last_metrics
                self.evaluator.summarize()
                # Now get the metrics that were just calculated
                console_metrics = getattr(self.evaluator, 'last_metrics', {})

            # Save predictions
            self.evaluator.save_preds()

            if console_metrics:
                print(f"📊 Captured mAP metrics for JSON output (full precision matches printed values)")
            else:
                print(f"⚠️  Warning: No mAP metrics available for JSON output")
        else:
            print("❌ No predictions to evaluate - all images failed prediction generation")
            console_metrics = {}

        # Save comprehensive results
        self.save_comprehensive_results(
            all_predictions, dataset, console_metrics, detailed_results
        )

        return {
            'processed_count': processed_count,
            'total_count': len(dataset),
            'predictions': all_predictions,
            'metrics': console_metrics,
            'detailed_results': detailed_results
        }

    def save_comprehensive_results(self, all_predictions, dataset, console_metrics, detailed_results):
        """Save evaluation results to file."""
        # Create results structure
        results = {
            "processing_summary": {
                "total_images_in_dataset": len(dataset),
                "successfully_processed": len(all_predictions),
                "failed_skipped": len(dataset) - len(all_predictions),
                "success_rate_percent": round((len(all_predictions) / len(dataset)) * 100, 1)
            },
            "evaluation_metrics": console_metrics,
            "successfully_processed_images": detailed_results,
            "timestamp": self.timestamp,  # Include timestamp in results for reference
            "output_directory": self.timestamped_output_dir
        }

        # Save results file in timestamped directory
        if self.timestamped_output_dir:
            dataset_name = getattr(self.args, 'dataset', 'unknown')
            results_file = os.path.join(self.timestamped_output_dir, f"{dataset_name}_evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Evaluation results saved: {results_file}")


def create_orchestrator(args):
    """Factory function to create and setup an orchestrator."""
    orchestrator = HOIEvaluationOrchestrator(args)
    orchestrator.setup_model()
    return orchestrator