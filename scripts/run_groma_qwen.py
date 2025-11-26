import os
import torch
import argparse
import re
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from groma.model.groma_qwen import GromaQwenModel, GromaQwenConfig
from groma.utils import disable_torch_init
"""
Example usage for grounding tasks:

Example 1: Person sitting on bench (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-24-nov-fixed/checkpoint-2500 \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
      --query "[grounding] Identify the following person and objects in the image: person sitting on bench and the bench" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 2

Example 2: Person playing with horse (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-24-nov-fixed/checkpoint-2500 \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000002.jpg \
      --query "[grounding] Identify the following person and objects in the image: person playing with horse and the horse" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 2

Example 3: Person holding cup (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-24-nov-fixed/checkpoint-2500 \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000003.jpg \
      --query "[grounding] Identify the following person and objects in the image: person holding cup and the cup" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 2

Example 4: Person riding bicycle (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-renew \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000004.jpg \
      --query "[grounding] Identify the following person and objects in the image: person riding bicycle and the bicycle" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 2

Example 5: Person talking to person (SWIG-HOI)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-renew \
      --image-file ../data/swig_hoi/images_512/talking_90.jpg \
      --query "[grounding] Identify the following person and objects in the image: person talking to person and the person" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 0

Example 6: Person carrying backpack (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-renew \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000005.jpg \
      --query "[grounding] Identify the following person and objects in the image: person carrying backpack and the backpack" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 0

Example 7: Multiple objects - Person, bench, and bag
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-renew \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
      --query "[grounding] Identify the following person and objects in the image: person sitting on bench, the bench, and any bag" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 0

Example 8: Person cutting cake (HICO-DET)
python scripts/run_groma_qwen.py \
      --model-name checkpoints/groma-qwen-stage3-hoi-qwen3vl-renew \
      --image-file ../data/hico_20160224_det/images/test2015/HICO_test2015_00000006.jpg \
      --query "[grounding] Identify the following person and objects in the image: person cutting cake and the cake" \
      --output-dir output-qwen-stage3-hoi \
      --gpu 0

"""
# Register custom model
AutoConfig.register("groma_qwen", GromaQwenConfig)
AutoModelForCausalLM.register(GromaQwenConfig, GromaQwenModel)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def draw_boxes(image, boxes, labels=None, output_path=None):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        # box is [x1, y1, x2, y2] (normalized 0-1)
        x1, y1, x2, y2 = box
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        if labels and i < len(labels):
            label = labels[i]
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1]-20, text_bbox[2], text_bbox[3]-20], fill="red")
            draw.text((x1, y1-20), label, fill="white", font=font)

    if output_path:
        image.save(output_path)
    return image

def parse_qwen_coordinates(text):
    """
    Parse Qwen-VL style coordinates from text.
    Expected format: (x1,y1),(x2,y2) where coords are 0-1000.
    Returns list of [x1, y1, x2, y2] normalized to 0-1.
    """
    # Regex for (x1,y1),(x2,y2)
    # Note: Qwen sometimes outputs <box>...</box> or just the coords.
    # We look for the coordinate pattern.
    pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
    matches = re.findall(pattern, text)
    
    boxes = []
    for match in matches:
        # Qwen coordinates are usually 0-1000
        x1, y1, x2, y2 = map(int, match)
        boxes.append([x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0])
    
    return boxes

def eval_model(args):
    # 1. Initialize and load config
    model_name = os.path.expanduser(args.model_name)

    print("="*80)
    print(f"Loading Groma-Qwen Model from {model_name}")
    print("="*80)

    print(f"\n1. Loading config from {model_name}...")
    config = GromaQwenConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ Config loaded successfully")
    print(f"   - Model type: {config.model_type}")
    print(f"   - New tokens: {config.num_new_token}")

    # 2. Load Processor
    # Note: The checkpoint contains the tokenizer with special tokens, but not the image processor
    # We load the tokenizer from checkpoint and image processor from base model
    print(f"\n2. Loading processor...")
    try:
        # Try loading from checkpoint first (has tokenizer with special tokens)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"   ✓ Processor loaded from checkpoint")
    except Exception as e:
        print(f"   ! Could not load full processor from checkpoint: {e}")
        print(f"   Loading from base model: {args.base_model_name}...")
        processor = AutoProcessor.from_pretrained(args.base_model_name, trust_remote_code=True)
        print(f"   ✓ Processor loaded from base model")

    # Get tokenizer - processor might be the tokenizer itself or have a tokenizer attribute
    tokenizer = processor if hasattr(processor, 'vocab_size') else processor.tokenizer
    print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")

    # Check if special tokens exist, if not add them
    from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS
    test_token = '<region>'
    test_id = tokenizer.convert_tokens_to_ids(test_token)
    if test_id == tokenizer.unk_token_id or test_id is None:
        print(f"   ! Special tokens not found, adding them...")
        num_added = tokenizer.add_tokens(
            list(DEFAULT_TOKENS.values()) + REGION_IDX_TOKENS,
            special_tokens=True
        )
        print(f"   ✓ Added {num_added} special tokens")
    else:
        print(f"   ✓ Special tokens already present (e.g., {test_token} -> {test_id})")

    # 3. Load Model with proper device mapping
    print(f"\n3. Loading GromaQwenModel weights...")
    disable_torch_init()
    
    # Set device based on GPU argument
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
        device_map = {"": device}
        dtype = torch.bfloat16
        print(f"   Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = "cpu"
        device_map = {"": device}
        dtype = torch.float32
        print(f"   CUDA not available, using CPU")
    
    model = GromaQwenModel.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    print(f"   ✓ Model loaded successfully")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")

    # 4. Initialize special token IDs
    print(f"\n4. Initializing special tokens...")
    model.init_special_token_id(tokenizer)
    print(f"   ✓ Special tokens initialized")
    print(f"   - Region token ID: {model.reg_token_id}")

    model.eval()
    print("\n" + "="*80)

    # 5. Prepare Input
    print("\n5. Preparing input...")
    image = load_image(args.image_file)

    # We need a full processor for image processing, not just tokenizer
    # If we only loaded tokenizer, load the full processor from a proper Qwen3-VL model
    if not hasattr(processor, 'image_processor'):
        print(f"   ! Processor doesn't have image_processor, loading full processor...")

        # Try loading from different sources in order
        # Prioritize local Qwen3-VL checkpoint, then checkpoint, then HuggingFace fallback
        qwen_base_models = [
            "Groma/checkpoints/Qwen3-VL-8B-Instruct",  # Local Qwen3-VL base model (PRIORITY)
            args.base_model_name,  # checkpoints/groma-qwen-stage3-hoi by default
            "checkpoints/Qwen3-VL-8B-Instruct",  # Alternative path
        ]

        full_processor = None
        for base_model in qwen_base_models:
            try:
                print(f"   Trying {base_model}...")
                full_processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
                if hasattr(full_processor, 'image_processor'):
                    print(f"   ✓ Loaded full processor from {base_model}")
                    break
                else:
                    print(f"   ! {base_model} lacks image_processor")
            except Exception as e:
                print(f"   ! Failed to load from {base_model}: {e}")

        if full_processor is None or not hasattr(full_processor, 'image_processor'):
            raise RuntimeError("Could not find a processor with image_processor. Please specify a valid Qwen3-VL base model.")

        # Create a combined processor-like object
        class CombinedProcessor:
            def __init__(self, tokenizer, base_processor):
                self.tokenizer = tokenizer
                self.image_processor = base_processor.image_processor
                self._base_processor = base_processor

            def apply_chat_template(self, *args, **kwargs):
                return self._base_processor.apply_chat_template(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self._base_processor(*args, **kwargs)

            def batch_decode(self, *args, **kwargs):
                return self.tokenizer.batch_decode(*args, **kwargs)

        processor = CombinedProcessor(tokenizer, full_processor)
        print(f"   ✓ Combined processor created with tokenizer from checkpoint")

    # Create messages matching training format (includes system message and initial greeting)
    # Training format includes:
    # 1. System: "Here is an image with region crops from it. Image: <image>. Regions: <region>."
    # 2. Assistant: "Thank you for the image! How can I assist you with it?"
    # 3. User: [grounding] query text
    from groma.constants import DEFAULT_TOKENS
    
    messages = [
        {
            "role": "system",
            "content": (
                f"Here is an image with region crops from it. "
                f"Image: {DEFAULT_TOKENS['image']}. "
                f"Regions: {DEFAULT_TOKENS['region']}."
            )
        },
        {
            "role": "assistant",
            "content": "Thank you for the image! How can I assist you with it?"
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.query},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    print(f"   ✓ Input prepared")

    # 6. Generate
    print("\n6. Generating response...")
    
    # Get EOS token ID from tokenizer (Qwen3-VL uses <|im_end|> token ID 151645)
    # Priority: <|im_end|> (Qwen3-VL) > eos_token_id > </s> (fallback)
    eos_token_id = None
    
    # First, try <|im_end|> token (Qwen3-VL)
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_token_id != tokenizer.unk_token_id:
        eos_token_id = im_end_token_id
        expected_eos = 151645
        if eos_token_id == expected_eos:
            print(f"   ✓ Using Qwen3-VL EOS token: {eos_token_id} (<|im_end|>)")
        else:
            print(f"   ⚠️  <|im_end|> token ID is {eos_token_id}, expected {expected_eos}")
    
    # Fallback to tokenizer's eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_token_id is not None:
            print(f"   Using tokenizer.eos_token_id: {eos_token_id}")
    
    # Final fallback: try </s> token
    if eos_token_id is None:
        eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        if eos_token_id != tokenizer.unk_token_id:
            print(f"   ⚠️  Using </s> token as fallback: {eos_token_id}")
        else:
            print(f"   ⚠️  WARNING: Could not find any EOS token!")
            eos_token_id = None
    
    # Ensure generation_config exists and is properly configured
    if not hasattr(model, 'generation_config') or model.generation_config is None:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
    
    # Set EOS and pad tokens
    if eos_token_id is not None:
        model.generation_config.eos_token_id = eos_token_id
        print(f"   - EOS token ID: {eos_token_id}")
    else:
        print(f"   ⚠️  Warning: EOS token ID is None!")
    
    # Set pad token (use EOS if pad is not available)
    pad_token_id = getattr(tokenizer, 'pad_token_id', None) or eos_token_id
    model.generation_config.pad_token_id = pad_token_id
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=eos_token_id,  # CRITICAL: Explicitly set EOS token
            pad_token_id=pad_token_id,
            do_sample=False,  # Greedy decoding for deterministic output
        )

    # 7. Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    
    print("\n" + "="*80)
    print("OUTPUT:")
    print(output_text)
    print("="*80)

    # 8. Parse and Visualize
    print("\n8. Parsing coordinates and visualizing...")
    boxes = parse_qwen_coordinates(output_text)
    
    if boxes:
        print(f"   ✓ Found {len(boxes)} bounding boxes")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.basename(args.image_file).split('.')[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_result.jpg")

        # Draw
        draw_boxes(image.copy(), boxes, output_path=output_path)
        print(f"   ✓ Saved visualization to {output_path}")

        # Save JSON
        json_path = os.path.join(args.output_dir, f"{base_name}_result.json")
        with open(json_path, 'w') as f:
            json.dump({
                "query": args.query,
                "response": output_text,
                "boxes": boxes
            }, f, indent=2)
        print(f"   ✓ Saved JSON to {json_path}")
    else:
        print("   ! No bounding boxes found in output")

    print("\n" + "="*80)
    print("✓ Inference completed successfully")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Groma-Qwen model inference on an image")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to Groma-Qwen checkpoint directory")
    parser.add_argument("--base-model-name", type=str,
                        default="checkpoints/groma-qwen-stage3-hoi",
                        help="Base model to load processor from (default: checkpoints/groma-qwen-stage3-hoi)")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to input image file")
    parser.add_argument("--query", type=str, required=True,
                        help="Query text to send to the model")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (unused, device_map='auto' is used)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")

    args = parser.parse_args()
    eval_model(args)
