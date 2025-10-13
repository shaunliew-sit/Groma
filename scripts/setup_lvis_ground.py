#!/usr/bin/env python3
"""
Script to set up LVIS-Ground evaluation by downloading only required COCO images.
This saves significant space by downloading ~4299 images instead of full train2017.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_lvis_annotation(output_dir):
    """Download lvis_test.json from HuggingFace"""
    print("Step 1: Downloading LVIS-Ground annotation file...")
    os.makedirs(output_dir, exist_ok=True)

    url = "https://huggingface.co/datasets/FoundationVision/groma_data/resolve/main/lvis_test.json"
    output_path = os.path.join(output_dir, "lvis_test.json")

    if os.path.exists(output_path):
        print(f"✓ {output_path} already exists")
        return output_path

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        sys.exit(1)


def diagnose_lvis_json(lvis_json_path):
    """Diagnose LVIS JSON structure and content"""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Analyzing LVIS JSON structure")
    print("="*60)

    with open(lvis_json_path, 'r') as f:
        data = json.load(f)

    print(f"Total images in LVIS test: {len(data['images'])}")

    # Show sample entries
    print("\nSample image entries:")
    for i in range(min(3, len(data['images']))):
        img = data['images'][i]
        print(f"\nImage {i+1}:")
        for key, value in img.items():
            if isinstance(value, str) and len(value) > 80:
                print(f"  {key}: {value[:80]}...")
            else:
                print(f"  {key}: {value}")

    # Check field availability
    has_coco_url = any('coco_url' in img for img in data['images'])
    has_file_name = any('file_name' in img for img in data['images'])
    has_flickr_url = any('flickr_url' in img for img in data['images'])

    print("\n" + "-"*60)
    print(f"Has 'coco_url' field: {has_coco_url}")
    print(f"Has 'file_name' field: {has_file_name}")
    print(f"Has 'flickr_url' field: {has_flickr_url}")

    # Analyze COCO splits if URLs exist
    if has_coco_url:
        splits = {}
        for img in data['images']:
            if 'coco_url' in img:
                url = img['coco_url']
                if 'val2017' in url:
                    splits['val2017'] = splits.get('val2017', 0) + 1
                elif 'train2017' in url:
                    splits['train2017'] = splits.get('train2017', 0) + 1
                elif 'test2017' in url:
                    splits['test2017'] = splits.get('test2017', 0) + 1
                elif 'unlisted' in url:
                    splits['unlisted2017'] = splits.get('unlisted2017', 0) + 1
                else:
                    splits['other'] = splits.get('other', 0) + 1

        print("\nImage distribution across COCO splits:")
        for split, count in sorted(splits.items()):
            print(f"  {split}: {count} images")

    print("="*60 + "\n")


def extract_required_images(lvis_json_path, diagnose=False):
    """Extract list of required COCO image filenames from lvis_test.json"""

    if diagnose:
        diagnose_lvis_json(lvis_json_path)

    print("Step 2: Extracting required image filenames...")

    with open(lvis_json_path, 'r') as f:
        data = json.load(f)

    # Extract filenames directly from the annotation
    # LVIS uses 'file_name' or 'coco_url' fields
    image_filenames = set()
    filename_to_url = {}  # Track which URL each filename came from

    for img in data['images']:
        if 'file_name' in img:
            # Use file_name directly
            filename = img['file_name']
            image_filenames.add(filename)
        elif 'coco_url' in img:
            # Extract filename from COCO URL
            url = img['coco_url']
            filename = url.split('/')[-1]
            image_filenames.add(filename)
            filename_to_url[filename] = url
        else:
            # Fallback: construct from ID
            filename = f"{img['id']:012d}.jpg"
            image_filenames.add(filename)

    image_filenames = sorted(list(image_filenames))
    print(f"✓ Found {len(image_filenames)} unique images required")

    # Show sample filenames for verification
    print(f"Sample filenames: {image_filenames[:3]}")

    # Show corresponding URLs if available
    if filename_to_url and image_filenames[:3]:
        print("Sample URLs:")
        for fname in image_filenames[:3]:
            if fname in filename_to_url:
                print(f"  {fname} -> {filename_to_url[fname]}")

    return image_filenames


def download_single_image(img_filename, output_dir):
    """Download a single COCO image, trying multiple splits and organizing by split"""
    # LVIS-Ground uses images from multiple COCO splits
    # Try them in order: val2017, train2017, test2017
    splits = ["val2017", "train2017", "test2017"]

    for split in splits:
        # Create split subdirectory
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        output_path = os.path.join(split_dir, img_filename)

        # Check if already exists in this split
        if os.path.exists(output_path):
            return True, img_filename, f"exists in {split}", split

        # Try to download from this split
        url = f"http://images.cocodataset.org/{split}/{img_filename}"
        try:
            urllib.request.urlretrieve(url, output_path)
            return True, img_filename, f"downloaded from {split}", split
        except Exception as e:
            # Continue to next split
            continue

    # If all splits failed
    return False, img_filename, "Not found in any COCO split", None


def download_required_images(image_filenames, output_dir, num_workers=8):
    """Download required COCO images in parallel, organized by split"""
    print(f"\nStep 3: Downloading {len(image_filenames)} COCO images...")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} parallel workers")
    print(f"Images will be organized by split: val2017/, train2017/, test2017/")

    os.makedirs(output_dir, exist_ok=True)

    downloaded = 0
    existed = 0
    failed = []
    split_counts = {"val2017": 0, "train2017": 0, "test2017": 0}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_single_image, fname, output_dir): fname
            for fname in image_filenames
        }

        with tqdm(total=len(image_filenames), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success, filename, status, split = future.result()
                if success:
                    if "exists" in status:
                        existed += 1
                    else:
                        downloaded += 1
                    # Count by split
                    if split and split in split_counts:
                        split_counts[split] += 1
                else:
                    failed.append((filename, status))
                pbar.update(1)

    print(f"\n✓ Download complete!")
    print(f"  - Downloaded: {downloaded}")
    print(f"  - Already existed: {existed}")
    print(f"\n  Images by split:")
    print(f"    • val2017: {split_counts['val2017']}")
    print(f"    • train2017: {split_counts['train2017']}")
    print(f"    • test2017: {split_counts['test2017']}")
    print(f"  - Failed: {len(failed)}")

    if failed:
        print("\nFailed downloads:")
        for fname, error in failed[:10]:
            print(f"  - {fname}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return len(failed) == 0


def verify_setup(lvis_json_path, images_dir):
    """Verify that all required images are available in their split folders"""
    print("\nStep 4: Verifying setup...")

    with open(lvis_json_path, 'r') as f:
        data = json.load(f)

    # Extract actual filenames
    required_filenames = set()
    for img in data['images']:
        if 'file_name' in img:
            required_filenames.add(img['file_name'])
        elif 'coco_url' in img:
            filename = img['coco_url'].split('/')[-1]
            required_filenames.add(filename)
        else:
            filename = f"{img['id']:012d}.jpg"
            required_filenames.add(filename)

    # Check in all possible split folders
    splits = ["val2017", "train2017", "test2017"]
    missing = []
    for filename in required_filenames:
        found = False
        for split in splits:
            filepath = os.path.join(images_dir, split, filename)
            if os.path.exists(filepath):
                found = True
                break
        if not found:
            missing.append(filename)

    if missing:
        print(f"✗ Missing {len(missing)} images:")
        for fname in sorted(missing)[:10]:
            print(f"  - {fname}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False
    else:
        print(f"✓ All {len(required_filenames)} required images are available")
        return True


def copy_from_coco_datasets(image_filenames, output_dir, coco_dirs):
    """Copy images from existing COCO dataset directories"""
    print(f"\nStep 3: Copying images from existing COCO datasets...")
    print(f"Output directory: {output_dir}")
    print(f"Source directories: {coco_dirs}")

    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    existed = 0
    missing = []

    for fname in tqdm(image_filenames, desc="Copying"):
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path):
            existed += 1
            continue

        # Try each COCO directory
        found = False
        for coco_dir in coco_dirs:
            source_path = os.path.join(coco_dir, fname)
            if os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, output_path)
                copied += 1
                found = True
                break

        if not found:
            missing.append(fname)

    print(f"\n✓ Copy complete!")
    print(f"  - Copied: {copied}")
    print(f"  - Already existed: {existed}")
    print(f"  - Missing: {len(missing)}")

    if missing:
        print(f"\n⚠️  Missing {len(missing)} images (not found in source directories)")
        print("Sample missing files:")
        for fname in missing[:10]:
            print(f"  - {fname}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Setup LVIS-Ground evaluation data")
    parser.add_argument("--output-dir", type=str, default="data_images/lvis_ground",
                        help="Output directory for LVIS annotation and images")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel download workers")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip image download (only download annotation)")
    parser.add_argument("--coco-dirs", type=str, nargs="+",
                        help="Paths to existing COCO image directories (e.g., coco/val2017 coco/train2017)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Show diagnostic information about LVIS JSON structure")
    args = parser.parse_args()

    print("="*60)
    print("LVIS-Ground Evaluation Setup")
    print("="*60)

    # Create directory structure
    annotation_dir = args.output_dir
    images_dir = os.path.join(args.output_dir, "images")

    # Step 1: Download annotation
    lvis_json_path = download_lvis_annotation(annotation_dir)

    # Step 2: Extract required images (with optional diagnosis)
    image_filenames = extract_required_images(lvis_json_path, diagnose=args.diagnose)

    # Step 3: Get images (download or copy)
    if not args.skip_download:
        if args.coco_dirs:
            # Copy from existing COCO directories
            copy_success = copy_from_coco_datasets(image_filenames, images_dir, args.coco_dirs)
        else:
            # Download from COCO servers
            download_success = download_required_images(image_filenames, images_dir, args.num_workers)

        # Step 4: Verify
        verify_success = verify_setup(lvis_json_path, images_dir)

        if verify_success:
            print("\n" + "="*60)
            print("✓ Setup complete! Ready to run evaluation.")
            print("="*60)
            print(f"\nAnnotation file: {lvis_json_path}")
            print(f"Images directory: {images_dir}")
            print(f"\nRun evaluation with:")
            print(f"python groma/eval/eval_lvis.py \\")
            print(f"    --model-name checkpoints/groma-7b-finetune \\")
            print(f"    --img-prefix {images_dir} \\")
            print(f"    --ann-file {lvis_json_path}")
        else:
            print("\n" + "="*60)
            print("⚠️  Setup incomplete - some images could not be downloaded")
            print("="*60)
            print("\nPossible reasons:")
            print("  1. Some images may have been removed from COCO servers")
            print("  2. Network issues or rate limiting")
            print("  3. Images may be in unlisted2017 or other special splits")
            print("\nYou can:")
            print("  • Try running the script again (may fix transient network errors)")
            print("  • Manually download COCO val2017 + train2017 full datasets")
            print("  • Proceed with available images (evaluation will skip missing ones)")
            sys.exit(1)
    else:
        print(f"\nSkipped image download. Run again without --skip-download to download images.")
        print(f"Total images needed: {len(image_filenames)}")


if __name__ == "__main__":
    main()
