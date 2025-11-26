import os
import copy
import torch
import argparse
import requests
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from transformers.image_transforms import center_to_corners_format
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates
"""
Example usage:
python scripts/run_groma.py \
      --model-name checkpoints/groma-qwen-stage3-hoi \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
      --query "[grounding] Identify the following person and objects in the image: person sitting on bench and the bench" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 1

"""


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def draw_box(box, image, index, output_dir, label=None, coords_text=None):
    """Draw bounding box on image with optional label and coordinates"""
    w, h = image.size
    box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    draw = ImageDraw.Draw(image)
    
    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw label if provided
    if label:
        # Draw background for text
        text_bbox = draw.textbbox((box[0], box[1] - 25), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((box[0], box[1] - 25), label, fill="white", font=font)
    
    # Draw coordinates if provided
    if coords_text:
        coords_y = box[1] - 45 if label else box[1] - 25
        coords_bbox = draw.textbbox((box[0], coords_y), coords_text, font=font)
        draw.rectangle(coords_bbox, fill="blue")
        draw.text((box[0], coords_y), coords_text, fill="white", font=font)
    
    output_file = os.path.join(output_dir, 'r{}.jpg'.format(index))
    image.save(output_file, "JPEG")
    return


def extract_coordinates_from_response(response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len, tokenizer):
    """Extract coordinate information and map them to objects in the response"""
    coordinates_info = []
    
    # Get the actual output tokens (excluding input)
    output_tokens = output_ids[0, input_token_len:]
    
    # Find all region token indices in the output
    selected_box_inds = []
    token_positions = []
    
    for pos, token_id in enumerate(output_tokens):
        if token_id.item() in box_idx_token_ids:
            box_idx = box_idx_token_ids.index(token_id.item())
            if box_idx < len(pred_boxes):
                selected_box_inds.append(box_idx)
                token_positions.append(pos)
    
    # Decode the response text and find region tokens
    decoded_tokens = tokenizer.convert_ids_to_tokens(output_tokens)
    
    # Extract context around each region token
    for i, (box_idx, token_pos) in enumerate(zip(selected_box_inds, token_positions)):
        box_coords = pred_boxes[box_idx].tolist()
        
        # Get surrounding context (up to 10 tokens before and after)
        context_start = max(0, token_pos - 10)
        context_end = min(len(decoded_tokens), token_pos + 10)
        context_tokens = decoded_tokens[context_start:context_end]
        context_text = tokenizer.convert_tokens_to_string(context_tokens)
        
        # Try to extract object description from context
        region_token = f"<r{box_idx}>"
        object_description = extract_object_description(response_text, region_token)
        
        # Debug: print the mapping for verification
        print(f"[DEBUG] Mapping {region_token} -> '{object_description}'")
        
        coordinates_info.append({
            'region_id': box_idx,
            'region_token': region_token,
            'coordinates': box_coords,  # [x1, y1, x2, y2] normalized
            'token_position': token_pos,
            'context': context_text.strip(),
            'object_description': object_description,
            'type': 'detected_region'
        })
    
    return coordinates_info, selected_box_inds


def debug_response_structure(response_text):
    """Debug function to show the structure of the response for better understanding"""
    print("\n[DEBUG] Response structure analysis:")
    
    # Find all <p> tags
    p_matches = re.findall(r'<p>([^<]+)</p>', response_text)
    if p_matches:
        print(f"Found {len(p_matches)} <p> tags:")
        for i, content in enumerate(p_matches):
            print(f"  <p>{i}: '{content}'")
    else:
        print("No <p> tags found")
    
    # Find all region tokens
    r_matches = re.findall(r'<r\d+>', response_text)
    if r_matches:
        print(f"Found region tokens: {r_matches}")
    
    print("[DEBUG] End response analysis\n")


def extract_object_description(response_text, region_token):
    """Extract object description from <p> tags by finding the correct <p> tag before each <roi><r#></roi> pattern"""
    
    # The correct pattern is: <p>description</p><roi><r#></roi>
    # We need to find the <p> tag that comes immediately before our specific region token
    
    # Look for the pattern: <p>content</p><roi><our_region_token></roi>
    escaped_token = re.escape(region_token)
    direct_pattern = rf'<p>([^<]+)</p><roi>{escaped_token}</roi>'
    
    match = re.search(direct_pattern, response_text)
    if match:
        return match.group(1).strip()
    
    # Alternative pattern: sometimes there might be spaces or variations
    # <p>content</p>\s*<roi>\s*<r#>\s*</roi>
    flexible_pattern = rf'<p>([^<]+)</p>\s*<roi>\s*{escaped_token}\s*</roi>'
    match = re.search(flexible_pattern, response_text)
    if match:
        return match.group(1).strip()
    
    # Another pattern: <p>content</p> followed by <r#> (without roi tags)
    simple_pattern = rf'<p>([^<]+)</p>.*?{escaped_token}'
    matches = list(re.finditer(simple_pattern, response_text))
    if matches:
        # Find the closest <p> tag before our region token
        region_pos = response_text.find(region_token)
        closest_match = None
        min_distance = float('inf')
        
        for match in matches:
            p_end_pos = match.end(1) + 4  # +4 for '</p>'
            distance = region_pos - p_end_pos
            if 0 < distance < min_distance:  # <p> tag should come before region token
                min_distance = distance
                closest_match = match
        
        if closest_match and min_distance < 50:  # Must be within reasonable distance
            return closest_match.group(1).strip()
    
    # Fallback to original pattern matching if no <p> tags found
    patterns = [
        rf'{re.escape(region_token)}\s+is\s+(?:a|an)?\s*([^<.!?]+)',
        rf'{re.escape(region_token)}\s+(?:shows?|depicts?)\s+(?:a|an)?\s*([^<.!?]+)',
        rf'([^<.!?]+)\s+{re.escape(region_token)}',
        rf'{re.escape(region_token)}\s*[,:]*\s*([^<.!?]+)',
        # Look for text between region token and next special token
        rf'{re.escape(region_token)}\s*[^<]*?([a-zA-Z][^<{{}}]*?)(?=<|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            desc = match.group(1).strip()
            # Clean up the description
            desc = re.sub(r'^(is|shows?|depicts?)\s+', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'^(a|an|the)\s+', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'[{}]', '', desc)  # Remove any remaining braces
            if len(desc) > 3:  # Only return meaningful descriptions
                return desc[:50]  # Limit length
    
    return f"Region {region_token}"


def format_coordinates(coords, original_width, original_height, coord_format="pixel"):
    """Format coordinates in different formats"""
    if coord_format == "pixel":
        return {
            'x1': int(coords[0] * original_width),
            'y1': int(coords[1] * original_height), 
            'x2': int(coords[2] * original_width),
            'y2': int(coords[3] * original_height),
            'width': int((coords[2] - coords[0]) * original_width),
            'height': int((coords[3] - coords[1]) * original_height),
            'center_x': int((coords[0] + coords[2]) * original_width / 2),
            'center_y': int((coords[1] + coords[3]) * original_height / 2)
        }
    else:  # normalized
        return {
            'x1': coords[0], 'y1': coords[1],
            'x2': coords[2], 'y2': coords[3],
            'width': coords[2] - coords[0],
            'height': coords[3] - coords[1],
            'center_x': (coords[0] + coords[2]) / 2,
            'center_y': (coords[1] + coords[3]) / 2
        }


def save_coordinates_json(coordinates_info, image_file, output_dir, original_width, original_height):
    """Save coordinate information to JSON file"""
    output_data = {
        'image_file': os.path.basename(image_file),
        'image_dimensions': {'width': original_width, 'height': original_height},
        'detected_regions': []
    }
    
    for coord_info in coordinates_info:
        pixel_coords = format_coordinates(coord_info['coordinates'], original_width, original_height, 'pixel')
        norm_coords = format_coordinates(coord_info['coordinates'], original_width, original_height, 'normalized')
        
        region_data = {
            'region_id': coord_info['region_id'],
            'region_token': coord_info['region_token'],
            'object_description': coord_info['object_description'],
            'coordinates_pixel': pixel_coords,
            'coordinates_normalized': norm_coords,
            'context': coord_info['context']
        }
        output_data['detected_regions'].append(region_data)
    
    json_file = os.path.join(output_dir, 'coordinates.json')
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return json_file


def eval_model(model_name, quant_type, image_file, query, device):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    kwargs = {}
    if quant_type == 'fp16':
        kwargs['torch_dtype'] = torch.float16
    elif quant_type == '8bit':
        kwargs['load_in_8bit'] = True
    elif quant_type == '4bit':
        int4_quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4'
        )
        kwargs = {'quantization_config': int4_quant_cfg}

    if quant_type == '8bit' or quant_type == '4bit':
        model = GromaModel.from_pretrained(model_name, **kwargs)
    else:
        model = GromaModel.from_pretrained(model_name, **kwargs).to(device)
    model.init_special_token_id(tokenizer)

    conversations = []
    instruct = "Here is an image with region crops from it. "
    instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
    instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))
    conversations.append((conv_templates['llava'].roles[0], query))
    conversations.append((conv_templates['llava'].roles[1], ''))
    prompt = conv_templates['llava'].get_prompt(conversations)

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).to(device)

    raw_image = load_image(image_file)
    raw_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].to(device)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type):
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config,
                # user-specified box input [x, y, w, h] (normalized)
                # refer_boxes=[torch.tensor([0.5874, 0.4748, 0.3462, 0.5059]).to(device).reshape(1, 4)]
            )
    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]
    pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
    pred_boxes = center_to_corners_format(pred_boxes)

    # Debug: Show all proposed boxes from detector
    print(f"\n{'='*60}")
    print(f"[DEBUG] DETECTOR PROPOSED BOXES (BEFORE filtering):")
    print(f"{'='*60}")
    print(f"Total regions proposed: {len(pred_boxes)}")
    for idx, box in enumerate(pred_boxes[:20]):  # Show first 20 boxes
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        area = w * h
        print(f"  r{idx}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}] "
              f"size=({w:.3f} × {h:.3f}) area={area:.3f}")
    if len(pred_boxes) > 20:
        print(f"  ... and {len(pred_boxes) - 20} more boxes")
    print(f"{'='*60}\n")

    # SMART FILTERING: Filter giant boxes BEFORE forward pass to prevent index remapping
    # Strategy: Remove boxes larger than 50% of image area (likely background/catch-all boxes)
    # This ensures model doesn't see giant boxes at inference time
    MAX_BOX_AREA_RATIO = 0.5  # Adjust this threshold based on your dataset

    box_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    valid_box_mask = box_areas < MAX_BOX_AREA_RATIO

    print(f"\n[SMART FILTER] Filtering boxes with area > {MAX_BOX_AREA_RATIO}")
    print(f"  Before: {len(pred_boxes)} boxes")
    print(f"  Removed giant boxes: {(~valid_box_mask).sum().item()} boxes")

    # Filter BEFORE passing to model
    pred_boxes = pred_boxes[valid_box_mask]
    print(f"  After: {len(pred_boxes)} boxes\n")

    box_idx_token_ids = model.box_idx_token_ids
    
    # Extract coordinates and object information
    coordinates_info, selected_box_inds = extract_coordinates_from_response(
        "", pred_boxes, box_idx_token_ids, output_ids, input_token_len, tokenizer
    )

    # Decode the response text for better object descriptions
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    response_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    response_text = response_text.strip()
    
    # Debug the response structure to understand <p> tag patterns
    debug_response_structure(response_text)
    
    # Re-extract coordinates with full response text for better object descriptions
    coordinates_info, selected_box_inds = extract_coordinates_from_response(
        response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len, tokenizer
    )

    # Create output directory
    output_dir = os.path.join(args.output_dir, image_file.split('.')[0].split('/')[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get original image dimensions
    original_width, original_height = raw_image.size
    
    # Save coordinate information to JSON
    json_file = save_coordinates_json(coordinates_info, image_file, output_dir, original_width, original_height)
    
    # Create annotated image with all boxes
    annotated_image = copy.deepcopy(raw_image)
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw all boxes on the main image
    for coord_info in coordinates_info:
        box = coord_info['coordinates']
        w, h = annotated_image.size
        pixel_box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        
        # Draw rectangle
        draw.rectangle(pixel_box, outline="red", width=3)
        
        # Prepare label text - use object description as primary label
        object_desc = coord_info['object_description']
        if object_desc.startswith('Region <r'):
            # If no meaningful description found, show the region token
            label = coord_info['region_token']
        else:
            # Use the actual object description from <p> tags
            label = object_desc
        coords_text = f"({int(pixel_box[0])},{int(pixel_box[1])}) {int(pixel_box[2]-pixel_box[0])}x{int(pixel_box[3]-pixel_box[1])}"
        
        # Draw label background and text
        text_y = pixel_box[1] - 45
        if text_y < 0:
            text_y = pixel_box[3] + 5
            
        label_bbox = draw.textbbox((pixel_box[0], text_y), label, font=font)
        draw.rectangle(label_bbox, fill="red")
        draw.text((pixel_box[0], text_y), label, fill="white", font=font)
        
        # Draw coordinates
        coords_y = text_y + 20
        coords_bbox = draw.textbbox((pixel_box[0], coords_y), coords_text, font=font)
        draw.rectangle(coords_bbox, fill="blue")
        draw.text((pixel_box[0], coords_y), coords_text, fill="white", font=font)
    
    # Save the main annotated image
    main_output_file = os.path.join(output_dir, 'annotated_image.jpg')
    annotated_image.save(main_output_file, "JPEG")

    # Create visualization of ALL filtered boxes (for debugging)
    all_boxes_image = copy.deepcopy(raw_image)
    draw_all = ImageDraw.Draw(all_boxes_image)

    # Draw all filtered boxes with their indices
    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box.numpy()
        pixel_box = [x1 * original_width, y1 * original_height,
                     x2 * original_width, y2 * original_height]

        # Use different colors for visual distinction
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
        color = colors[idx % len(colors)]

        # Draw rectangle
        draw_all.rectangle(pixel_box, outline=color, width=2)

        # Draw region label
        label = f"r{idx}"
        area = (x2 - x1) * (y2 - y1)
        label_with_area = f"{label} ({area:.2f})"

        text_y = pixel_box[1] - 20
        if text_y < 0:
            text_y = pixel_box[3] + 5

        label_bbox = draw_all.textbbox((pixel_box[0], text_y), label_with_area, font=font)
        draw_all.rectangle(label_bbox, fill=color)
        draw_all.text((pixel_box[0], text_y), label_with_area, fill="white", font=font)

    all_boxes_file = os.path.join(output_dir, 'all_proposed_regions.jpg')
    all_boxes_image.save(all_boxes_file, "JPEG")

    # Create side-by-side comparison
    comparison_width = original_width * 2
    comparison_height = original_height
    comparison_image = Image.new('RGB', (comparison_width, comparison_height), color='white')

    # Paste all boxes on left, selected boxes on right
    comparison_image.paste(all_boxes_image, (0, 0))
    comparison_image.paste(annotated_image, (original_width, 0))

    # Add labels
    draw_comparison = ImageDraw.Draw(comparison_image)
    draw_comparison.text((10, 10), "ALL PROPOSED REGIONS", fill="red", font=font)
    draw_comparison.text((original_width + 10, 10), "MODEL SELECTION", fill="red", font=font)

    comparison_file = os.path.join(output_dir, 'comparison.jpg')
    comparison_image.save(comparison_file, "JPEG")

    # Create individual region images (existing functionality)
    for i, coord_info in enumerate(coordinates_info):
        box = coord_info['coordinates']
        img_copy = copy.deepcopy(raw_image)
        
        # Format coordinates for display
        pixel_coords = format_coordinates(box, original_width, original_height, 'pixel')
        coords_text = f"({pixel_coords['x1']},{pixel_coords['y1']}) {pixel_coords['width']}x{pixel_coords['height']}"
        
        # Use meaningful description for individual images too
        individual_label = coord_info['object_description'] if not coord_info['object_description'].startswith('Region <r') else coord_info['region_token']
        
        draw_box(
            box, img_copy, coord_info['region_id'], output_dir,
            label=individual_label,
            coords_text=coords_text
        )
    
    # Print response and coordinate information
    print("=" * 60)
    print("MODEL RESPONSE:")
    print(response_text)
    print("=" * 60)
    print("DETECTED COORDINATES:")
    
    for coord_info in coordinates_info:
        pixel_coords = format_coordinates(coord_info['coordinates'], original_width, original_height, 'pixel')
        norm_coords = format_coordinates(coord_info['coordinates'], original_width, original_height, 'normalized')
        
        # Display object description prominently
        object_desc = coord_info['object_description']
        if object_desc.startswith('Region <r'):
            print(f"\n{coord_info['region_token']}: [No description found]")
        else:
            print(f"\n{coord_info['region_token']}: {object_desc}")
        
        print(f"  Pixel coordinates: ({pixel_coords['x1']}, {pixel_coords['y1']}) to ({pixel_coords['x2']}, {pixel_coords['y2']})")
        print(f"  Size: {pixel_coords['width']} x {pixel_coords['height']} pixels")
        print(f"  Center: ({pixel_coords['center_x']}, {pixel_coords['center_y']})")
        print(f"  Normalized: ({norm_coords['x1']:.3f}, {norm_coords['y1']:.3f}) to ({norm_coords['x2']:.3f}, {norm_coords['y2']:.3f})")
        if coord_info['context']:
            print(f"  Context: {coord_info['context'][:100]}...")
    
    print(f"\n=" * 60)
    print(f"OUTPUTS SAVED TO: {output_dir}")
    print(f"- Annotated image: {main_output_file}")
    print(f"- All proposed regions: {all_boxes_file}")
    print(f"- Side-by-side comparison: {comparison_file}")
    print(f"- Individual regions: r{{N}}.jpg files")
    print(f"- Coordinate data: {json_file}")
    print(f"=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--quant_type", type=str, default='none')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    # Set GPU device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: cuda:{args.gpu}")
        print(f"GPU Name: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    model_name = os.path.expanduser(args.model_name)
    if args.image_dir is not None:
        image_files = sorted(os.listdir(args.image_dir))
        for image_file in image_files:
            image_file = os.path.join(args.image_dir, image_file)
            eval_model(model_name, args.quant_type, image_file, args.query, device)
    elif args.image_file is not None:
        eval_model(model_name, args.quant_type, args.image_file, args.query, device)
    else:
        print("Please specify image file or image directory.")