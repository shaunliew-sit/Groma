import os
import json
import argparse
import subprocess
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def check_java_installed():
    """Check if Java is installed and accessible."""
    try:
        subprocess.run(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=str, default="refcocog.json")
    parser.add_argument("--result-dir", type=str, default="refcocog_eval_output")
    parser.add_argument("--skip-java-check", action="store_true",
                        help="Skip Java check (use if you know Java is installed)")
    args = parser.parse_args()

    # Check for Java installation (required by COCO caption evaluator)
    if not args.skip_java_check and not check_java_installed():
        print("=" * 80)
        print("ERROR: Java is not installed or not in PATH")
        print("=" * 80)
        print("\nThe COCO caption evaluation toolkit requires Java for tokenization.")
        print("\nTo fix this issue, install Java:")
        print("  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y default-jdk")
        print("  - CentOS/RHEL:   sudo yum install -y java-11-openjdk")
        print("  - macOS:         brew install openjdk@11")
        print("  - Docker:        Add 'RUN apt-get update && apt-get install -y default-jdk' to Dockerfile")
        print("\nAlternatively, install Java in conda environment:")
        print("  conda install -c conda-forge openjdk")
        print("\nAfter installation, verify with: java -version")
        print("=" * 80)
        sys.exit(1)

    results = []
    result_files = [f for f in os.listdir(args.result_dir) if f.endswith('.json') and 'all.json' not in f and 'temp_ann.json' not in f]

    print(f"Found {len(result_files)} result file(s) to process:")
    for result_file in result_files:
        file_path = f"{args.result_dir}/{result_file}"
        print(f"  - {result_file}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Handle both single dict and list of dicts
                if isinstance(data, dict):
                    results.append(data)
                elif isinstance(data, list):
                    results.extend(data)
                else:
                    print(f"    Warning: Unexpected data type in {result_file}: {type(data)}")
        except json.JSONDecodeError as e:
            print(f"    Error: Failed to parse JSON in {result_file}: {e}")
            continue
        except Exception as e:
            print(f"    Error: Failed to read {result_file}: {e}")
            continue

    if not results:
        print("\nError: No valid results found!")
        print(f"Please check that {args.result_dir} contains result JSON files.")
        sys.exit(1)

    print(f"\nTotal results loaded: {len(results)}")

    # Deduplicate results by image_id (keep last occurrence)
    results_map = dict()
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            print(f"Warning: Skipping non-dict result at index {i}: {type(result)}")
            continue
        if 'image_id' not in result:
            print(f"Warning: Result at index {i} missing 'image_id': {result}")
            continue
        key = result['image_id']
        results_map[key] = i

    result_inds = results_map.values()
    results = [results[i] for i in result_inds]

    print(f"Unique images after deduplication: {len(results)}")

    all_results_file = f"{args.result_dir}/all.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f)

    # Load annotation file and ensure it has required COCO fields
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    # Add missing COCO fields if they don't exist
    if 'info' not in ann_data:
        ann_data['info'] = {'description': 'Visual Genome Test Set'}
    if 'licenses' not in ann_data:
        ann_data['licenses'] = []

    # Create a temporary annotation file with required fields
    temp_ann_file = f"{args.result_dir}/temp_ann.json"
    with open(temp_ann_file, 'w') as f:
        json.dump(ann_data, f)

    coco = COCO(temp_ann_file)
    coco_result = coco.loadRes(all_results_file)
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    # Clean up temporary annotation file
    if os.path.exists(temp_ann_file):
        os.remove(temp_ann_file)
