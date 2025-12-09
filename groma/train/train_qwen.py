# [GROMA-QWEN V2] Training script for GromaQwenModel (Native Architecture)
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# Key Changes from V1:
# ====================
# - No new token addition (uses native Qwen3VL tokens only)
# - No DINOv2 image processor (uses Qwen3VL processor)
# - Simplified freeze logic (only LLM freeze, no perceiver)
#
# Training Strategy:
# ==================
# Stage 2 Only (recommended):
#   --freeze_llm True
#   Trains: ROI Align module + Image-to-Text Bridge
#   Frozen: Qwen3VL LLM + Vision Encoder
#
# Dataset Selection:
# ==================
# Set TASK_TYPE environment variable:
#   TASK_TYPE=referring  - Train on referring task only
#   TASK_TYPE=grounding  - Train on grounding task only
#   TASK_TYPE=both       - Train on both tasks (default)
#
# Example:
# ========
# TASK_TYPE=referring deepspeed --num_gpus=8 \
#     -m groma.train.train_qwen \
#     --llm checkpoints/Qwen3-VL-8B-Instruct \
#     --output_dir checkpoints/groma-qwen-v2-stage2 \
#     --dataset_config groma/data/configs/vl_train_stage2_qwen3vl_native.py \
#     --freeze_llm True \
#     --per_device_train_batch_size 4 \
#     --num_train_epochs 4 \
#     --learning_rate 2e-4

import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoProcessor, AutoTokenizer

from groma.model.groma_qwen import GromaQwenConfig, GromaQwenModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForHybridDataset
from groma.train.groma_trainer import GromaTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)
    # Kept for backward compatibility, not used in V2
    perceiver: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/data/configs/vl_train_stage2_qwen3vl_native.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_llm: bool = field(default=True)  # Default True for Stage 2
    freeze_vl_bridge: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=2048)
    group_by_data_source: Optional[bool] = field(default=True)
    # Custom learning rate settings (used by GromaTrainer)
    use_custom_lr: bool = field(default=False)
    custom_lr: float = field(default=1e-4)
    custom_lr_params: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of parameter name patterns for custom LR"}
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Determine tokenizer source
    tokenizer_path = model_args.model_name_or_path if model_args.model_name_or_path else model_args.llm
    
    print("=" * 60)
    print("[GROMA-QWEN V2] Training with Native Qwen3VL Architecture")
    print("=" * 60)
    
    # Load Tokenizer
    print(f"\n[1] Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    
    # Verify Qwen3-VL tokenizer
    name_or_path = getattr(tokenizer, 'name_or_path', '') or ''
    if 'qwen2' in name_or_path.lower():
        print(f"⚠️  WARNING: Detected Qwen2-VL tokenizer: {name_or_path}")
        print("   This training script requires Qwen3-VL tokenizer!")
    elif 'qwen3' in name_or_path.lower():
        print(f"✓ Verified Qwen3-VL tokenizer")
    
    # Verify native bbox tokens exist
    box_start_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
    box_end_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
    
    if box_start_id == tokenizer.unk_token_id:
        raise ValueError("Native token <|box_start|> not found in tokenizer")
    if box_end_id == tokenizer.unk_token_id:
        raise ValueError("Native token <|box_end|> not found in tokenizer")
    
    print(f"✓ Native bbox tokens verified:")
    print(f"  - <|box_start|>: {box_start_id}")
    print(f"  - <|box_end|>: {box_end_id}")
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Initialize Model
    print(f"\n[2] Initializing model...")
    
    if model_args.model_name_or_path:
        # Resume from checkpoint
        print(f"   Loading from checkpoint: {model_args.model_name_or_path}")
        model = GromaQwenModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )
    elif model_args.llm:
        # Initialize from base LLM
        print(f"   Initializing from base LLM: {model_args.llm}")
        
        config = GromaQwenConfig.from_pretrained(model_args.llm, trust_remote_code=True)
        config.num_new_token = 0  # V2: No new tokens
        
        model = GromaQwenModel.from_pretrained(
            model_args.llm,
            config=config,
            trust_remote_code=True
        )
    else:
        raise ValueError("Must specify either model_name_or_path or llm")
    
    # Initialize special token IDs (verifies native tokens)
    model.init_special_token_id(tokenizer)
    print(f"✓ Model initialized")
    
    # Freeze/Unfreeze Logic
    print(f"\n[3] Configuring trainable parameters...")
    
    # Freeze LLM (always for Stage 2)
    if training_args.freeze_llm:
        model.model.requires_grad_(False)
        print(f"   ✓ LLM frozen (Stage 2 mode)")
    else:
        print(f"   ✓ LLM trainable (Stage 3 mode)")
    
    # Freeze Image-to-Text Bridge
    if training_args.freeze_vl_bridge:
        model.img_txt_bridge.requires_grad_(False)
        print(f"   ✓ Image-to-Text Bridge frozen")
    else:
        print(f"   ✓ Image-to-Text Bridge trainable")
    
    # Region Encoder is always trainable
    model.region_encoder.requires_grad_(True)
    print(f"   ✓ Region Encoder (ROI Align) trainable")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load Processor (for image processing)
    print(f"\n[4] Loading image processor...")
    processor = AutoProcessor.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    img_processor = processor.image_processor
    print(f"✓ Image processor loaded")
    
    # Build Datasets
    print(f"\n[5] Building datasets...")
    print(f"   Config: {data_args.dataset_config}")
    
    train_datasets = build_multi_datasets(
        data_args.dataset_config,
        tokenizer=tokenizer,
        img_processor=img_processor
    )
    
    print(f"✓ Dataset loaded with {len(train_datasets)} samples")
    
    # Data Collator
    data_collator = DataCollatorForHybridDataset(tokenizer)
    
    # Trainer
    print(f"\n[6] Starting training...")
    trainer = GromaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=data_collator
    )
    
    # Resume from checkpoint if exists
    checkpoint_path = pathlib.Path(training_args.output_dir)
    if list(checkpoint_path.glob("checkpoint-*")):
        print(f"   Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save model
    print(f"\n[7] Saving model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    
    # Save processor
    processor.save_pretrained(training_args.output_dir)
    print(f"✓ Training complete!")


if __name__ == "__main__":
    train()
