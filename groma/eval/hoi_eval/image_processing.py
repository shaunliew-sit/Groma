"""Image processing utilities for HOI evaluation with Groma model."""

import re
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers.image_transforms import center_to_corners_format

from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


def load_image(image_file):
    """Load image from file or URL"""
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_hoi_response(model, tokenizer, vis_processor, image, prompt):
    """Generate response using Groma model"""

    # Build conversation
    conversations = []

    # Initial setup
    instruct = "Here is an image with region crops from it. "
    instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
    instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))

    # Add the HOI question
    conversations.append((conv_templates['llava'].roles[0], prompt))
    conversations.append((conv_templates['llava'].roles[1], ''))

    prompt_full = conv_templates['llava'].get_prompt(conversations)
    inputs = tokenizer([prompt_full])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config,
            )

    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]

    # Extract coordinates
    pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
    pred_boxes = center_to_corners_format(pred_boxes)
    box_idx_token_ids = model.box_idx_token_ids

    # Decode response
    response_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    response_text = response_text.strip()

    # Extract coordinates
    coordinates_info = extract_coordinates_from_response(
        response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len
    )

    return response_text, coordinates_info


def generate_hoi_response_batch(model, tokenizer, vis_processor, images, prompt, batch_size=8):
    """Generate responses using Groma model for a batch of images"""

    # Build conversation template (same for all images)
    conversations = []
    instruct = "Here is an image with region crops from it. "
    instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
    instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))
    conversations.append((conv_templates['llava'].roles[0], prompt))
    conversations.append((conv_templates['llava'].roles[1], ''))

    prompt_full = conv_templates['llava'].get_prompt(conversations)

    # Process images in batches
    all_results = []
    num_images = len(images)

    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_images = images[batch_start:batch_end]
        current_batch_size = len(batch_images)

        # Prepare batched inputs
        batch_inputs = tokenizer([prompt_full] * current_batch_size)
        input_ids = torch.as_tensor(batch_inputs.input_ids).cuda()

        # Stack images for batch processing
        batched_images = torch.cat(batch_images, dim=0)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                outputs = model.generate(
                    input_ids,
                    images=batched_images,
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=1024,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    generation_config=model.generation_config,
                )

        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        box_idx_token_ids = model.box_idx_token_ids

        # Process each image in the batch
        for i in range(current_batch_size):
            # Extract coordinates for this image
            pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][i].cpu()
            pred_boxes = center_to_corners_format(pred_boxes)

            # Decode response for this image
            response_text = tokenizer.batch_decode(output_ids[i:i+1, input_token_len:], skip_special_tokens=False)[0]
            response_text = response_text.strip()

            # Extract coordinates for this image
            coordinates_info = extract_coordinates_from_response_batch(
                response_text, pred_boxes, box_idx_token_ids, output_ids[i:i+1], input_token_len
            )

            all_results.append((response_text, coordinates_info))

    return all_results


def extract_coordinates_from_response_batch(response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len):
    """Extract coordinate information from grounded response for batch processing"""
    coordinates_info = []
    output_tokens = output_ids[0, input_token_len:]
    selected_box_inds = []

    for pos, token_id in enumerate(output_tokens):
        if token_id.item() in box_idx_token_ids:
            box_idx = box_idx_token_ids.index(token_id.item())
            if box_idx < len(pred_boxes):
                selected_box_inds.append(box_idx)

    for box_idx in selected_box_inds:
        box_coords = pred_boxes[box_idx].tolist()
        region_token = f"<r{box_idx}>"
        object_description = extract_object_description(response_text, region_token)

        coordinates_info.append({
            'region_id': box_idx,
            'region_token': region_token,
            'coordinates': box_coords,
            'object_description': object_description,
            'type': 'detected_region'
        })

    return coordinates_info


def extract_coordinates_from_response(response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len):
    """Extract coordinate information from grounded response"""
    coordinates_info = []
    output_tokens = output_ids[0, input_token_len:]
    selected_box_inds = []

    for pos, token_id in enumerate(output_tokens):
        if token_id.item() in box_idx_token_ids:
            box_idx = box_idx_token_ids.index(token_id.item())
            if box_idx < len(pred_boxes):
                selected_box_inds.append(box_idx)

    for box_idx in selected_box_inds:
        box_coords = pred_boxes[box_idx].tolist()
        region_token = f"<r{box_idx}>"
        object_description = extract_object_description(response_text, region_token)

        coordinates_info.append({
            'region_id': box_idx,
            'region_token': region_token,
            'coordinates': box_coords,
            'object_description': object_description,
            'type': 'detected_region'
        })

    return coordinates_info


def extract_object_description(response_text, region_token):
    """Extract object description from grounded response"""
    escaped_token = re.escape(region_token)

    patterns = [
        rf'<p>([^<]+)</p><roi>{escaped_token}</roi>',
        rf'<p>([^<]+)</p>\s*<roi>\s*{escaped_token}\s*</roi>',
        rf'<p>([^<]+)</p>.*?{escaped_token}'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).strip()

    return f"Region {region_token}"