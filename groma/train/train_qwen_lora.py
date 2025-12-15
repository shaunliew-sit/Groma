# [GROMA-QWEN] LoRA Training script for GromaQwenModel
# Part of: Groma Qwen3VL Referring Task Implementation
# Based on: groma/train/train_qwen.py
#
# This script implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA
# for the Groma-Qwen model. LoRA is applied ONLY to the Qwen3VL LLM component,
# while Groma-specific components (region_encoder, img_txt_bridge, new_input_embs)
# remain fully trainable.
#
# Benefits of LoRA:
# =================
# - Prevents catastrophic forgetting of base LLM knowledge
# - Trains only ~0.5% of LLM parameters (50M vs 8B)
# - Enables higher learning rates (1e-4 vs 2e-6)
# - Reduces memory usage and training time
#
# Architecture:
# =============
# - vis_encoder (DINOv2): Frozen (unchanged)
# - region_encoder: Fully trainable (task-critical)
# - img_txt_bridge: Fully trainable (task-critical)
# - new_input_embs: Fully trainable (special tokens)
# - model (Qwen3VL LLM): LoRA adapters on attention + MLP layers
#
# Usage:
# ======
# Called by: scripts/vl_finetune_referring_lora.sh
#
# Key LoRA Parameters:
# - lora_r: Rank of LoRA matrices (default: 64)
# - lora_alpha: Scaling factor (default: 128)
# - lora_dropout: Dropout for regularization (default: 0.05)
# - lora_target_modules: Which layers to adapt (attention + MLP)
#

import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import AutoImageProcessor, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType, inject_adapter_in_model

from groma.model.groma_qwen import GromaQwenConfig, GromaQwenModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForHybridDataset
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS
from groma.train.groma_trainer import GromaTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)
    perceiver: Optional[str] = field(default=None)  # Path to DINOv2 or ignored
    nms_thres: Optional[float] = field(default=0.6)
    box_score_thres: Optional[float] = field(default=0.)
    max_region_num: Optional[int] = field(default=100)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/datasets/dataset_configs.py')


@dataclass
class LoraArguments:
    """LoRA-specific arguments for PEFT fine-tuning."""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for training"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank - controls adapter capacity"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha - scaling factor (typically 2x rank)"}
    )
    lora_dropout: float = field(
        default=0.05,
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


def prepare_model_for_lora(model, lora_args: LoraArguments):
    """
    Apply LoRA to Qwen LLM component while keeping Groma components fully trainable.
    
    Architecture after LoRA:
    - vis_encoder (DINOv2): Frozen (no change)
    - region_encoder: Fully trainable (requires_grad=True)
    - img_txt_bridge: Fully trainable (requires_grad=True)
    - new_input_embs: Fully trainable (requires_grad=True)
    - model (Qwen3VL): LoRA adapters only
    
    Args:
        model: GromaQwenModel instance
        lora_args: LoRA configuration arguments
    
    Returns:
        model: Model with LoRA applied to Qwen LLM
    """
    print("\n" + "="*60)
    print("Applying LoRA to Qwen3VL LLM")
    print("="*60)
    
    # 1. Ensure Groma components are fully trainable
    print("\n1. Setting Groma components to fully trainable...")
    model.region_encoder.requires_grad_(True)
    model.img_txt_bridge.requires_grad_(True)
    model.new_input_embs.requires_grad_(True)
    print("   ✓ region_encoder: requires_grad=True")
    print("   ✓ img_txt_bridge: requires_grad=True")
    print("   ✓ new_input_embs: requires_grad=True")
    
    # 2. vis_encoder stays frozen (already set in GromaQwenModel.__init__)
    model.vis_encoder.requires_grad_(False)
    print("   ✓ vis_encoder (DINOv2): Frozen")
    
    # 3. Parse target modules
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    print(f"\n2. LoRA Configuration:")
    print(f"   - Rank (r): {lora_args.lora_r}")
    print(f"   - Alpha: {lora_args.lora_alpha}")
    print(f"   - Dropout: {lora_args.lora_dropout}")
    print(f"   - Bias: {lora_args.lora_bias}")
    print(f"   - Target modules: {target_modules}")
    
    # 4. Create LoRA config
    # Note: We don't use task_type because we'll inject adapters directly
    # This avoids issues with prepare_inputs_for_generation on inner models
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
    )
    
    # 5. Apply LoRA to Qwen LLM
    # We use inject_adapter_in_model instead of get_peft_model to avoid
    # issues with prepare_inputs_for_generation on the inner Qwen3VLModel.
    # inject_adapter_in_model adds LoRA layers directly without wrapping.
    print("\n3. Applying LoRA to Qwen3VL LLM...")
    
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
    print("\n4. Trainable Parameters Summary:")
    
    # Count parameters by component
    def count_params(module, name):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    components = [
        ("vis_encoder (DINOv2)", model.vis_encoder),
        ("region_encoder", model.region_encoder),
        ("img_txt_bridge", model.img_txt_bridge),
        ("new_input_embs", model.new_input_embs),
        ("model (Qwen3VL + LoRA)", model.model),
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


def merge_lora_weights_into_model(model, lora_args: LoraArguments):
    """
    Merge LoRA weights into base model and replace LoRA layers with standard Linear layers.
    
    This is a proper "merge and unload" operation that:
    1. Computes merged weights: W' = W_base + (B @ A) * scaling
    2. Replaces LoRA Linear layers with standard nn.Linear containing merged weights
    3. Results in a standard model that can be saved/loaded without PEFT
    
    LoRA computes: output = base_weight @ x + (lora_B @ lora_A) @ x * scaling
    Merged: new_weight = base_weight + lora_B @ lora_A * scaling
    
    Args:
        model: GromaQwenModel with injected LoRA layers
        lora_args: LoRA configuration for getting scaling factor
    
    Returns:
        model: Model with merged weights (LoRA layers replaced with standard Linear)
    """
    import torch.nn as nn
    from peft.tuners.lora.layer import Linear as LoraLinear
    
    print("\n" + "="*60)
    print("Merging LoRA weights and unloading adapters")
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
                # Fallback for different PEFT versions
                base_layer = lora_layer
            
            # Get base weight and bias
            base_weight = base_layer.weight.data.clone()
            base_bias = base_layer.bias.data.clone() if base_layer.bias is not None else None
            
            # Get LoRA weights
            if hasattr(lora_layer, 'lora_A') and hasattr(lora_layer, 'lora_B'):
                # Get the actual weight tensors
                if hasattr(lora_layer.lora_A, 'default'):
                    # ModuleDict format: lora_A['default']
                    lora_A = lora_layer.lora_A['default'].weight
                    lora_B = lora_layer.lora_B['default'].weight
                elif isinstance(lora_layer.lora_A, nn.ModuleDict):
                    # Try to get the first adapter
                    adapter_name = list(lora_layer.lora_A.keys())[0]
                    lora_A = lora_layer.lora_A[adapter_name].weight
                    lora_B = lora_layer.lora_B[adapter_name].weight
                else:
                    # Direct module format
                    lora_A = lora_layer.lora_A.weight
                    lora_B = lora_layer.lora_B.weight
                
                # Use layer's scaling if available, otherwise use computed scaling
                if hasattr(lora_layer, 'scaling'):
                    if isinstance(lora_layer.scaling, dict):
                        layer_scaling = lora_layer.scaling.get('default', scaling)
                    else:
                        layer_scaling = lora_layer.scaling
                else:
                    layer_scaling = scaling
                
                # Merge: W' = W + B @ A * scaling
                # lora_A shape: (r, in_features)
                # lora_B shape: (out_features, r)
                delta = (lora_B @ lora_A) * layer_scaling
                merged_weight = base_weight + delta
                merge_count += 1
            else:
                # No LoRA weights, just use base weight
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

    print("\n" + "="*60)
    print("Groma-Qwen LoRA Training")
    print("="*60)
    print(f"LoRA Enabled: {lora_args.use_lora}")
    print(f"Model Path: {model_args.model_name_or_path or model_args.llm}")
    print(f"Output Dir: {training_args.output_dir}")
    print("="*60 + "\n")

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
        config = GromaQwenConfig.from_pretrained(model_args.llm, trust_remote_code=True)
        config.num_new_token = num_new_token
        
        model = GromaQwenModel.from_pretrained(
            model_args.llm,
            config=config,
            trust_remote_code=True
        )
    else:
        raise ValueError("Should specify either model_name_or_path or llm.")
    
    # Initialize special token IDs in model
    model.init_special_token_id(tokenizer)
    
    # Image Processor (for DINOv2)
    try:
        vis_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    except:
        if model_args.perceiver:
            vis_processor = AutoImageProcessor.from_pretrained(model_args.perceiver)
        else:
            raise ValueError("Could not load AutoImageProcessor for facebook/dinov2-large")

    # Apply LoRA if enabled
    if lora_args.use_lora:
        model = prepare_model_for_lora(model, lora_args)
    else:
        # Fallback to standard freeze/unfreeze logic (like train_qwen.py)
        print("LoRA disabled, using standard fine-tuning...")
        if training_args.freeze_perceiver:
            model.vis_encoder.requires_grad_(False)
        if training_args.freeze_llm:
            model.model.requires_grad_(False)
        if training_args.freeze_vl_bridge:
            model.img_txt_bridge.requires_grad_(False)

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

    # Resume from checkpoint if exists
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Merge LoRA weights before saving (so the saved model is compatible with existing eval scripts)
    if lora_args.use_lora:
        print("\nMerging LoRA weights before saving...")
        model = merge_lora_weights_into_model(model, lora_args)

    # Save model
    trainer.save_model()
    trainer.save_state()
    vis_processor.save_pretrained(training_args.output_dir)
    
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
            }, f, indent=2)
        print(f"\n✓ LoRA config saved to {lora_config_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoint saved to: {training_args.output_dir}")
    if lora_args.use_lora:
        print("\n✓ LoRA weights have been merged into the model.")
        print("  You can evaluate directly with the existing eval script:")
        print(f"  python groma/eval/eval_hico_action_referring_groma_qwen.py \\")
        print(f"      --model-name {training_args.output_dir} \\")
        print(f"      --img-prefix ../data/hico_20160224_det/images/test2015 \\")
        print(f"      --ann-file groma_data/benchmarks/hico_action_referring_test.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()

