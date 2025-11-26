# Training script for GromaQwenModel
# Adapted from groma/train/train.py

import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import AutoImageProcessor, AutoTokenizer

from groma.model.groma_qwen import GromaQwenConfig, GromaQwenModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForHybridDataset
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS
from groma.train.groma_trainer import GromaTrainer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)
    perceiver: Optional[str] = field(default=None) # Path to DINOv2 or ignored
    nms_thres: Optional[float] = field(default=0.6)
    box_score_thres: Optional[float] = field(default=0.)
    max_region_num: Optional[int] = field(default=100)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/datasets/dataset_configs.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_llm: bool = field(default=False)
    freeze_perceiver: bool = field(default=True)
    freeze_vl_bridge: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=512)
    use_custom_lr: bool = field(default=False)
    custom_lr_params: tuple = field(default=('perceiver', 'llm'))
    custom_lr: float = field(default=2e-5)
    group_by_data_source: Optional[bool] = field(default=True)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load Tokenizer (Qwen3-VL)
    tokenizer_path = model_args.model_name_or_path if model_args.model_name_or_path else model_args.llm
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True 
    )
    
    # Verify Qwen3-VL tokenizer (not Qwen2-VL)
    name_or_path = getattr(tokenizer, 'name_or_path', '') or ''
    if isinstance(name_or_path, str):
        name_lower = name_or_path.lower()
        if 'qwen2' in name_lower or 'qwen2-vl' in name_lower:
            print(f"⚠️  WARNING: Detected Qwen2-VL tokenizer: {name_or_path}")
            print("   This training script requires Qwen3-VL tokenizer!")
            print("   Please ensure tokenizer_path points to Qwen3-VL-8B-Instruct")
        elif 'qwen3' in name_lower or 'qwen3-vl' in name_lower:
            print(f"✓ Verified Qwen3-VL tokenizer: {name_or_path}")
    
    # Verify apply_chat_template method exists (required for Qwen3VL template)
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError(
            "Tokenizer must have apply_chat_template method for Qwen3VL template. "
            f"Tokenizer path: {tokenizer_path}. "
            "Please ensure you're using Qwen3-VL tokenizer from Groma/checkpoints/Qwen3-VL-8B-Instruct"
        )
    print("✓ Tokenizer has apply_chat_template method")
    
    # Verify EOS token ID (should be 151645 for Qwen3-VL)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_token_id != tokenizer.unk_token_id:
        expected_eos = 151645
        if im_end_token_id == expected_eos:
            print(f"✓ EOS token ID verified: {im_end_token_id} (<|im_end|>)")
        else:
            print(f"⚠️  WARNING: <|im_end|> token ID is {im_end_token_id}, expected {expected_eos}")
    else:
        print("⚠️  WARNING: Could not find <|im_end|> token in vocabulary")
    
    # Add special tokens
    # Qwen tokenizer might already have some, but we add Groma tokens
    # Check if tokens exist first to avoid duplicates (handled by tokenizer.add_tokens usually)
    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + REGION_IDX_TOKENS, special_tokens=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_TOKENS['pad']

    # Initialize Model
    if model_args.model_name_or_path:
        # Resume from checkpoint or load finetuned model
        model = GromaQwenModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )
    elif model_args.llm:
        # Initialize from LLM + Perceiver (Stage 2)
        # We assume model_args.llm points to Qwen base
        
        # Config
        # We can pass vis_encoder_cfg if we want to customize DINOv2, 
        # but GromaQwenConfig defaults to facebook/dinov2-large.
        # If model_args.perceiver is provided and is a path to DINOv2 weights, 
        # GromaQwenModel handles loading it if it matches standard HF format, 
        # OR we can let GromaQwenModel load from hub.
        
        # For now, we initialize config with num_new_token
        config = GromaQwenConfig.from_pretrained(model_args.llm, trust_remote_code=True)
        config.num_new_token = num_new_token
        
        # Initialize Model
        model = GromaQwenModel.from_pretrained(
            model_args.llm,
            config=config,
            trust_remote_code=True
        )
        
        # If perceiver path is provided and valid, we could try to load it, 
        # but GromaQwenModel loads DINOv2 internally. 
        # If user wants to use local DINOv2, they should modify GromaQwenModel or we rely on HF cache.
        
    else:
        raise ValueError("Should specify either model_name_or_path or llm.")

    # Resize embeddings for new tokens
    # model.resize_token_embeddings(len(tokenizer))
    
    # Initialize special token IDs in model
    model.init_special_token_id(tokenizer)
    
    # Image Processor (for DINOv2)
    # We use standard DINOv2 ImageProcessor
    try:
        vis_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    except:
        # Fallback or use perceiver path if it points to something with preprocessor
        if model_args.perceiver:
             vis_processor = AutoImageProcessor.from_pretrained(model_args.perceiver)
        else:
            raise ValueError("Could not load AutoImageProcessor for facebook/dinov2-large")

    # Freeze/Unfreeze logic
    # GromaQwenModel has .vis_encoder (perceiver) and .model (llm)
    
    if training_args.freeze_perceiver:
        model.vis_encoder.requires_grad_(False)
    
    if training_args.freeze_llm:
        model.model.requires_grad_(False) # Qwen LLM
    
    if training_args.freeze_vl_bridge:
        model.img_txt_bridge.requires_grad_(False)
        
    # Always train new embeddings and region encoder (unless specified otherwise, but usually trained)
    # model.region_encoder.requires_grad_(True) # Default is True
    # model.new_input_embs.requires_grad_(True) # Default is True

    # Build Datasets
    train_datasets = build_multi_datasets(
        data_args.dataset_config,
        tokenizer=tokenizer,
        img_processor=vis_processor
    )
    
    data_collator = DataCollatorForHybridDataset(tokenizer)

    trainer = GromaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=data_collator
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model()
    trainer.save_state()
    vis_processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
