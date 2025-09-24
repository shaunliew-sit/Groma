"""Main evaluation orchestrator for HOI detection evaluation."""

import os
import json
import datetime
import torch
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
            # HICO evaluator expects: anno_file, output_dir, zero_shot_type, ignore_non_interaction
            anno_file = os.path.join(self.args.data_root, 'annotations', 'test_hico_ann.json')
            self.evaluator = HICOEvaluator(anno_file, output_dir, 'uc0', True)
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

    def evaluate_dataset(self, dataset_type, data_root, max_images=None, batch_size=1):
        """Evaluate an entire dataset."""
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

        all_predictions = {}
        detailed_results = {}
        processed_count = 0

        # Process images
        for i, data_item in enumerate(dataset):
            try:
                image_id = data_item['image_id']
                image_path = data_item['image_path']

                print(f"\n[{i+1}/{len(dataset)}] Processing image {image_id}: {os.path.basename(image_path)}")

                # Use single image evaluation
                result = self.evaluate_single_image(image_path, dataset_type, data_root)

                # Store predictions for final evaluation
                if result['predictions']:
                    all_predictions[image_id] = result['predictions']

                # Store detailed results
                detailed_results[image_id] = {
                    'file_name': os.path.basename(image_path),
                    'triplets_found': len(result['triplets']),
                    'predictions_count': len(result['predictions']),
                    'gt_count': len(result['ground_truth']),
                    'metrics': result['metrics'],
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
                        for triplet in result['triplets']
                    ]
                }

                processed_count += 1

            except Exception as e:
                print(f"❌ Error processing image {data_item['image_id']}: {str(e)}")
                continue

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

        # Update evaluator with all predictions (original script approach)
        if all_predictions:
            self.evaluator.update(all_predictions)

            # Calculate final metrics
            self.evaluator.accumulate()
            console_metrics = self.evaluator.summarize()
        else:
            print("No predictions to evaluate")
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