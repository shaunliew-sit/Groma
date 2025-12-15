# [GROMA-QWEN V2] Stage 3 LoRA Training script for GromaQwenModel (Native Architecture)
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# This script implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA
# for the Groma-Qwen V2 model. LoRA is applied ONLY to the Qwen3VL LLM component,
# while Groma-specific components (region_encoder, img_txt_bridge) remain fully trainable.
#
# Key Differences from V1 LoRA (train_qwen_lora.py):
# ==================================================
# - Uses GromaQwenModel (V2) instead of V1 with DINOv2
# - No vis_encoder (DINOv2) - uses Qwen3VL vision encoder
# - No new_input_embs - uses native Qwen3VL tokens
# - Vision encoder accessed via model.visual (frozen)
# - LLM accessed via model.model (LoRA applied here)
#
# Architecture for Stage 3:
# =========================
# - visual (Qwen3VL Vision): Frozen (no change)
# - region_encoder (ROI Align): Fully trainable (task-critical)
# - img_txt_bridge: Fully trainable (task-critical)
# - model (Qwen3VL LLM): LoRA adapters only
#
# Benefits of LoRA for Stage 3:
# =============================
# - Aligns output format with training data expectations
# - Prevents catastrophic forgetting of base LLM knowledge
# - Trains only ~0.5% of LLM parameters (50M vs 8B)
# - Enables higher learning rates than full fine-tuning
#
# Task Types (via TASK_TYPE env var):
# ===================================
# - referring: Train on action recognition (output format alignment)
# - grounding: Train on object localization (instruction understanding)
# - both: Train on combined tasks (comprehensive improvement)
#
# Usage:
# ======
# Called by: scripts/vl_train_stage3_qwen3vl_native_lora.sh

import os
import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoProcessor, AutoTokenizer

from peft import LoraConfig, inject_adapter_in_model

from groma.model.groma_qwen import GromaQwenConfig, GromaQwenModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForHybridDataset
from groma.train.groma_trainer import GromaTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/data/configs/vl_train_stage2_qwen3vl_native.py')


@dataclass
class LoraArguments:
    """LoRA-specific arguments for PEFT fine-tuning."""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for training"}
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank - controls adapter capacity"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha - scaling factor (typically 2x rank)"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout for regularization"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type: 'none', 'all', or 'lora_only'"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vl_bridge: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=2048)
    group_by_data_source: Optional[bool] = field(default=True)
    # Custom learning rate settings (required by GromaTrainer)
    use_custom_lr: bool = field(default=False)
    custom_lr: float = field(default=1e-4)
    custom_lr_params: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of parameter name patterns for custom LR"}
    )


def prepare_model_for_lora_v2(model, lora_args: LoraArguments):
    """
    Apply LoRA to Qwen LLM component while keeping Groma V2 components fully trainable.
    
    Architecture after LoRA (V2):
    - visual (Qwen3VL Vision): Frozen (no change)
    - region_encoder: Fully trainable (requires_grad=True)
    - img_txt_bridge: Fully trainable (requires_grad=True)
    - model (Qwen3VL LLM): LoRA adapters only
    
    Args:
        model: GromaQwenModel (V2) instance
        lora_args: LoRA configuration arguments
    
    Returns:
        model: Model with LoRA applied to Qwen LLM
    """
    print("\n" + "="*60)
    print("[GROMA-QWEN V2] Applying LoRA to Qwen3VL LLM")
    print("="*60)
    
    # 1. Freeze vision encoder (Qwen3VL visual)
    model.visual.requires_grad_(False)
    print("\n1. Freezing Qwen3VL Vision Encoder...")
    print("   ✓ visual: Frozen")
    
    # 2. Ensure Groma V2 components are fully trainable
    print("\n2. Setting Groma V2 components to fully trainable...")
    model.region_encoder.requires_grad_(True)
    model.img_txt_bridge.requires_grad_(True)
    print("   ✓ region_encoder (ROI Align): requires_grad=True")
    print("   ✓ img_txt_bridge: requires_grad=True")
    
    # 3. Parse target modules
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    print(f"\n3. LoRA Configuration:")
    print(f"   - Rank (r): {lora_args.lora_r}")
    print(f"   - Alpha: {lora_args.lora_alpha}")
    print(f"   - Dropout: {lora_args.lora_dropout}")
    print(f"   - Bias: {lora_args.lora_bias}")
    print(f"   - Target modules: {target_modules}")
    
    # 4. Create LoRA config
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
    )
    
    # 5. Apply LoRA to Qwen3VL LLM (model.model)
    print("\n4. Applying LoRA to Qwen3VL LLM...")
    
    # First, freeze the base Qwen model weights (we only train LoRA adapters)
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Inject LoRA adapters into the model
    inject_adapter_in_model(lora_config, model.model, adapter_name="default")
    
    # Enable gradients for LoRA parameters
    lora_params = 0
    for name, param in model.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params += param.numel()
    
    print(f"   ✓ LoRA adapters injected successfully")
    print(f"   ✓ LoRA parameters: {lora_params:,}")
    
    # 6. Print trainable parameter summary
    print("\n5. Trainable Parameters Summary:")
    
    # Count parameters by component
    def count_params(module, name):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    components = [
        ("visual (Qwen3VL Vision)", model.visual),
        ("region_encoder (ROI Align)", model.region_encoder),
        ("img_txt_bridge", model.img_txt_bridge),
        ("model (Qwen3VL LLM + LoRA)", model.model),
    ]
    
    total_all = 0
    trainable_all = 0
    for name, module in components:
        total, trainable = count_params(module, name)
        total_all += total
        trainable_all += trainable
        pct = 100 * trainable / total if total > 0 else 0
        print(f"   {name}:")
        print(f"      Total: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")
    
    print(f"\n   TOTAL:")
    print(f"      All params: {total_all:,}")
    print(f"      Trainable: {trainable_all:,} ({100*trainable_all/total_all:.2f}%)")
    print("="*60 + "\n")
    
    return model


def merge_lora_weights_into_model_v2(model, lora_args: LoraArguments):
    """
    Merge LoRA weights into base model and replace LoRA layers with standard Linear layers.
    
    This is a proper "merge and unload" operation for V2 architecture.
    
    Args:
        model: GromaQwenModel (V2) with injected LoRA layers
        lora_args: LoRA configuration for getting scaling factor
    
    Returns:
        model: Model with merged weights (LoRA layers replaced with standard Linear)
    """
    import torch.nn as nn
    from peft.tuners.lora.layer import Linear as LoraLinear
    
    print("\n" + "="*60)
    print("[GROMA-QWEN V2] Merging LoRA weights and unloading adapters")
    print("="*60)
    
    # Calculate scaling factor
    scaling = lora_args.lora_alpha / lora_args.lora_r
    print(f"   LoRA scaling factor: {scaling} (alpha={lora_args.lora_alpha} / r={lora_args.lora_r})")
    
    merge_count = 0
    replace_count = 0
    
    # Collect all LoRA layers and their parent modules
    lora_layers_to_replace = []
    
    def find_lora_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoraLinear):
                lora_layers_to_replace.append((module, name, child, full_name))
            else:
                find_lora_layers(child, full_name)
    
    find_lora_layers(model.model)
    print(f"   Found {len(lora_layers_to_replace)} LoRA layers to merge and replace")
    
    # Merge and replace each LoRA layer
    for parent, attr_name, lora_layer, full_name in lora_layers_to_replace:
        with torch.no_grad():
            # Get base layer
            if hasattr(lora_layer, 'base_layer'):
                base_layer = lora_layer.base_layer
            else:
                base_layer = lora_layer
            
            # Get base weight and bias
            base_weight = base_layer.weight.data.clone()
            base_bias = base_layer.bias.data.clone() if base_layer.bias is not None else None
            
            # Get LoRA weights
            if hasattr(lora_layer, 'lora_A') and hasattr(lora_layer, 'lora_B'):
                # Get the actual weight tensors
                if hasattr(lora_layer.lora_A, 'default'):
                    lora_A = lora_layer.lora_A['default'].weight
                    lora_B = lora_layer.lora_B['default'].weight
                elif isinstance(lora_layer.lora_A, nn.ModuleDict):
                    adapter_name = list(lora_layer.lora_A.keys())[0]
                    lora_A = lora_layer.lora_A[adapter_name].weight
                    lora_B = lora_layer.lora_B[adapter_name].weight
                else:
                    lora_A = lora_layer.lora_A.weight
                    lora_B = lora_layer.lora_B.weight
                
                # Use layer's scaling if available
                if hasattr(lora_layer, 'scaling'):
                    if isinstance(lora_layer.scaling, dict):
                        layer_scaling = lora_layer.scaling.get('default', scaling)
                    else:
                        layer_scaling = lora_layer.scaling
                else:
                    layer_scaling = scaling
                
                # Merge: W' = W + B @ A * scaling
                delta = (lora_B @ lora_A) * layer_scaling
                merged_weight = base_weight + delta
                merge_count += 1
            else:
                merged_weight = base_weight
            
            # Create standard Linear layer with merged weights
            out_features, in_features = merged_weight.shape
            new_linear = nn.Linear(in_features, out_features, bias=(base_bias is not None))
            new_linear.weight.data = merged_weight
            if base_bias is not None:
                new_linear.bias.data = base_bias
            
            # Move to same device and dtype as original
            new_linear = new_linear.to(device=base_weight.device, dtype=base_weight.dtype)
            
            # Replace the LoRA layer with standard Linear
            setattr(parent, attr_name, new_linear)
            replace_count += 1
    
    print(f"   ✓ Merged {merge_count} LoRA weight matrices")
    print(f"   ✓ Replaced {replace_count} LoRA layers with standard Linear layers")
    print("="*60 + "\n")
    
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Get task type from environment
    task_type = os.environ.get('TASK_TYPE', 'both')
    
    print("\n" + "="*60)
    print("[GROMA-QWEN V2] Stage 3 LoRA Training")
    print("="*60)
    print(f"Task Type: {task_type}")
    print(f"LoRA Enabled: {lora_args.use_lora}")
    print(f"Model Path: {model_args.model_name_or_path or model_args.llm}")
    print(f"Output Dir: {training_args.output_dir}")
    print("="*60 + "\n")

    # Determine tokenizer source
    tokenizer_path = model_args.model_name_or_path if model_args.model_name_or_path else model_args.llm
    
    # Load Tokenizer
    print(f"[1] Loading tokenizer from: {tokenizer_path}")
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
        # Load from Stage 2 checkpoint
        print(f"   Loading from checkpoint: {model_args.model_name_or_path}")
        model = GromaQwenModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )
    elif model_args.llm:
        # Initialize from base LLM (not recommended for Stage 3)
        print(f"   ⚠️  Initializing from base LLM: {model_args.llm}")
        print(f"   WARNING: For Stage 3, you should load from a Stage 2 checkpoint!")
        
        config = GromaQwenConfig.from_pretrained(model_args.llm, trust_remote_code=True)
        config.num_new_token = 0
        
        model = GromaQwenModel.from_pretrained(
            model_args.llm,
            config=config,
            trust_remote_code=True
        )
    else:
        raise ValueError("Must specify either model_name_or_path or llm")
    
    # Initialize special token IDs
    model.init_special_token_id(tokenizer)
    print(f"✓ Model initialized")
    
    # Apply LoRA if enabled
    if lora_args.use_lora:
        model = prepare_model_for_lora_v2(model, lora_args)
    else:
        # Fallback to standard fine-tuning
        print("\n[3] LoRA disabled, using standard fine-tuning...")
        model.visual.requires_grad_(False)
        print("   ✓ visual: Frozen")
        
        if training_args.freeze_vl_bridge:
            model.img_txt_bridge.requires_grad_(False)
            print("   ✓ img_txt_bridge: Frozen")
        else:
            model.img_txt_bridge.requires_grad_(True)
            print("   ✓ img_txt_bridge: Trainable")
        
        model.region_encoder.requires_grad_(True)
        print("   ✓ region_encoder: Trainable")
        
        model.model.requires_grad_(True)
        print("   ✓ model (Qwen3VL LLM): Trainable")
    
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
    print(f"   Task Type: {task_type}")
    
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
    
    # Merge LoRA weights before saving
    if lora_args.use_lora:
        print("\n[7] Merging LoRA weights before saving...")
        model = merge_lora_weights_into_model_v2(model, lora_args)
    
    # Save model
    print(f"\n[8] Saving model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    
    # Save processor
    processor.save_pretrained(training_args.output_dir)
    
    # Save LoRA config for reference
    if lora_args.use_lora:
        import json
        lora_config_path = pathlib.Path(training_args.output_dir) / "lora_config.json"
        with open(lora_config_path, 'w') as f:
            json.dump({
                "lora_r": lora_args.lora_r,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "lora_target_modules": lora_args.lora_target_modules,
                "lora_bias": lora_args.lora_bias,
                "merged_before_save": True,
                "task_type": task_type,
                "architecture": "groma_qwen_v2_native",
            }, f, indent=2)
        print(f"✓ LoRA config saved to {lora_config_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoint saved to: {training_args.output_dir}")
    if lora_args.use_lora:
        print("\n✓ LoRA weights have been merged into the model.")
        print("  You can evaluate directly with the existing eval scripts:")
        if task_type == 'referring':
            print(f"\n  # Referring evaluation:")
            print(f"  bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 {training_args.output_dir}")
        elif task_type == 'grounding':
            print(f"\n  # Grounding evaluation:")
            print(f"  bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 {training_args.output_dir}")
        else:
            print(f"\n  # Referring evaluation:")
            print(f"  bash scripts/run_hico_action_referring_groma_qwen_v2_eval.sh 0 {training_args.output_dir}")
            print(f"\n  # Grounding evaluation:")
            print(f"  bash scripts/run_hico_ground_groma_qwen_v2_eval.sh 0 {training_args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()

