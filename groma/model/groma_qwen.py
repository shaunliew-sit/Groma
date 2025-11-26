import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Dinov2Model,
    Dinov2Config,
    PretrainedConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from groma.model.roi_align import MLVLROIQueryModule
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS

class GromaQwenConfig(Qwen3VLConfig):
    model_type = "groma_qwen"

    def __init__(
        self,
        vis_encoder_cfg=None,
        num_new_token=0,
        roi_align_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Fix for Qwen3VLConfig not propagating rope_scaling to text_config when manually initialized
        if hasattr(self, "rope_scaling") and self.rope_scaling is not None:
            if hasattr(self, "text_config") and self.text_config is not None:
                if not hasattr(self.text_config, "rope_scaling") or self.text_config.rope_scaling is None:
                    self.text_config.rope_scaling = self.rope_scaling
            if hasattr(self, "vision_config") and self.vision_config is not None:
                if not hasattr(self.vision_config, "rope_scaling") or self.vision_config.rope_scaling is None:
                    self.vision_config.rope_scaling = self.rope_scaling
        
        # Default to facebook/dinov2-large config if not provided
        if vis_encoder_cfg is None:
            # We can't load from string here easily without connection, 
            # but we can set default dict for dinov2-large
            self.vis_encoder_cfg = Dinov2Config(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                mlp_ratio=4,
                hidden_act="gelu",
                image_size=518, # DINOv2 large default
                patch_size=14
            )
        elif isinstance(vis_encoder_cfg, dict):
            self.vis_encoder_cfg = Dinov2Config(**vis_encoder_cfg)
        elif isinstance(vis_encoder_cfg, Dinov2Config):
            self.vis_encoder_cfg = vis_encoder_cfg
        else:
            self.vis_encoder_cfg = vis_encoder_cfg

        self.num_new_token = num_new_token
        self.roi_align_cfg = roi_align_cfg if roi_align_cfg is not None else {}
        
        # Explicitly set loss_type to suppress warning
        if not hasattr(self, "loss_type") or self.loss_type is None:
            self.loss_type = "ForCausalLMLoss"


class GromaQwenModel(Qwen3VLForConditionalGeneration):
    config_class = GromaQwenConfig

    def __init__(self, config: GromaQwenConfig):
        super().__init__(config)

        # 1. Initialize Region Encoder Backbone (DINOv2) - Frozen
        # Load DINOv2 from config (weights will be loaded from checkpoint via from_pretrained)
        print("Initializing DINOv2 encoder from config...")
        self.vis_encoder = Dinov2Model(config.vis_encoder_cfg)
        self.vis_encoder.requires_grad_(False) # Freeze DINOv2

        # 2. Initialize Region Encoder Head (ROI Align + Fusion) - Trained
        # DINOv2 Large hidden size = 1024
        # Use the actual hidden size from the loaded encoder to avoid mismatch
        image_embed_dim = self.vis_encoder.config.hidden_size 
        image_embed_dim = self.vis_encoder.config.hidden_size 
        # Qwen3VLConfig might store hidden_size in text_config or directly
        text_embed_dim = getattr(config, "hidden_size", None)
        if text_embed_dim is None:
            if hasattr(config, "text_config"):
                text_embed_dim = getattr(config.text_config, "hidden_size", 4096)
            else:
                text_embed_dim = 4096 # Fallback default for 7B/8B models
        
        # MLVLROIQueryModule takes (embed_dims, out_dims, num_levels)
        # Note: out_dims in MLVLROIQueryModule init seems unused in forward? 
        # It returns query_feats from MlvlRoIExtractor.
        # MlvlRoIExtractor outputs 'out_channels' which is set to embed_dims (1024).
        # Then MlvlRoIExtractor has a linear updims 1024 -> 4096.
        # We need to check if 4096 is hardcoded or passed.
        self.region_encoder = MLVLROIQueryModule(
            embed_dims=image_embed_dim, 
            out_dims=4096, # This parameter is passed to MlvlRoIExtractor
            num_levels=3
        )
        
        # Bridge to project region features to LLM space
        # The region encoder (MlvlRoIExtractor) outputs features of dimension 4096.
        # We project this to the LLM's hidden size (text_embed_dim).
        self.img_txt_bridge = nn.Sequential(
            nn.Linear(4096, text_embed_dim),
            nn.GELU(),
            nn.Linear(text_embed_dim, text_embed_dim),
        )
        
        # 3. Special Tokens Embeddings - Trained
        # Handles <r0>...<r99>, <region>, etc.
        self.new_input_embs = nn.Embedding(config.num_new_token, text_embed_dim)
        
        # Initialize with average of existing embeddings
        # Use get_input_embeddings() for compatibility
        embed_tokens = self.model.get_input_embeddings()
        if embed_tokens is not None:
            with torch.no_grad():
                input_embeds = embed_tokens.weight.data
                input_embeds_avg = input_embeds.mean(dim=0, keepdim=True)
                self.new_input_embs.weight.data.copy_(input_embeds_avg.expand(config.num_new_token, -1))

        # Token IDs
        self.reg_token_id = None
        self.refer_feat_token_id = None
        self.box_idx_token_ids = None

    def init_special_token_id(self, tokenizer):
        # Ensure these tokens are in tokenizer
        self.reg_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS['region'])
        self.refer_feat_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS['rfeat'])
        self.box_idx_token_ids = tokenizer.convert_tokens_to_ids(REGION_IDX_TOKENS)

    def _compute_custom_embeddings(self, input_ids):
        # Combine original Qwen embeddings with new special token embeddings
        ori_vocab_size = self.config.vocab_size
        
        # Mask for new tokens (assuming they are appended > vocab_size)
        # Note: Qwen3VL might have large vocab. We assume new tokens are added AT THE END.
        # We need to know the original vocab size BEFORE adding new tokens to know the boundary.
        # Or we can check if ID >= config.vocab_size (if config isn't updated yet).
        # Safer: use a known offset or check against current model embedding size.
        
        ori_embed_tokens = self.model.get_input_embeddings()
        num_ori_tokens = ori_embed_tokens.num_embeddings
        
        mask = input_ids >= num_ori_tokens
        
        # Get original embeddings
        ori_ids = input_ids.masked_fill(mask, 0)
        input_embeddings = ori_embed_tokens(ori_ids)
        
        # Get new embeddings
        new_ids = input_ids - num_ori_tokens
        new_ids = new_ids.masked_fill(~mask, 0)
        new_input_embeddings = self.new_input_embs(new_ids)
        
        # Combine
        input_embeddings[mask] = new_input_embeddings[mask]
        return input_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        # Groma specific args
        region_images=None,
        refer_boxes=None,
        **kwargs,
    ):
        """
        Override prepare_inputs_for_generation to preserve region_images and refer_boxes
        during generation. These parameters are needed for referring tasks.
        
        Without this override, region_images and refer_boxes would be lost after the first
        generation step, causing referring tasks to fail during multi-step generation.
        """
        # Call parent's prepare_inputs_for_generation to handle standard Qwen3VL parameters
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        
        # Preserve Groma-specific parameters
        # These are needed for referring tasks where region features must be injected
        # at <refer_feat> token positions during each generation step
        if region_images is not None:
            model_inputs["region_images"] = region_images
        if refer_boxes is not None:
            model_inputs["refer_boxes"] = refer_boxes
        
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Qwen3-VL specific args
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Groma specific args
        region_images: Optional[torch.Tensor] = None, # Images for DINOv2 (B, C, H, W)
        refer_boxes: Optional[List[torch.Tensor]] = None, # Boxes for ROI Align (B, N, 4)
        **kwargs
    ):
        
        # We cannot pass inputs_embeds to Qwen3VL because it requires input_ids for RoPE,
        # but forbids passing both.
        # Solution: Use a forward hook on embed_tokens to inject our custom embeddings
        # while passing masked input_ids to Qwen.
        
        region_features_proj = None
        
        # 1. Process Regions (if present)
        # Check for None AND non-empty tensors (empty tensors should be treated as None)
        # This ensures grounding tasks (which have no refer_boxes) skip region processing
        has_regions = (
            region_images is not None 
            and refer_boxes is not None
            and (isinstance(region_images, torch.Tensor) and region_images.shape[0] > 0)
            and (isinstance(refer_boxes, list) and len(refer_boxes) > 0 and refer_boxes[0].shape[0] > 0)
        )
        
        if has_regions:
            # Get model dtype for consistency
            model_dtype = next(self.region_encoder.parameters()).dtype
            
            # Ensure region_images are in the right dtype (DINOv2 can handle both)
            # Convert to float32 for DINOv2 (it expects float32), then convert outputs
            region_images_float = region_images.float() if region_images.dtype != torch.float32 else region_images
            
            # region_images: (B, C, H, W)
            with torch.no_grad():
                # DINOv2 Forward (outputs float32)
                vis_outputs = self.vis_encoder(region_images_float, output_hidden_states=True)
                
                # Extract Last 3 Layers
                mlvl_feats = vis_outputs.hidden_states[-3:]
                # Remove CLS token (index 0) -> (B, L-1, D)
                mlvl_feats = [f[:, 1:] for f in mlvl_feats]
            
            # Convert DINOv2 features to model dtype (bfloat16) to match region encoder weights
            # DINOv2 outputs float32, but region encoder expects bfloat16 when model is in bfloat16
            mlvl_feats = [f.to(dtype=model_dtype) for f in mlvl_feats]
            
            # Coordinate Alignment (0-1000 -> 0-1)
            norm_refer_boxes = [box.float() / 1000.0 for box in refer_boxes]
            
            # Region Encoder
            region_features = self.region_encoder(mlvl_feats, norm_refer_boxes)
            
            # Project to LLM dim
            region_features_flat = torch.cat(region_features, dim=0) 
            region_features_proj = self.img_txt_bridge(region_features_flat)

        # 2. Prepare Masked Input IDs and Hook
        # We mask new tokens in input_ids so embed_tokens doesn't crash
        ori_vocab_size = self.model.get_input_embeddings().num_embeddings
        
        if input_ids is not None:
            # Identify new tokens
            new_token_mask = input_ids >= ori_vocab_size
            
            # Mask them with 0 (or any valid ID) for the base model call
            masked_input_ids = input_ids.masked_fill(new_token_mask, 0)
            
            # Define Hook
            def embedding_hook(module, inputs, output):
                # output is the embedding of masked_input_ids
                # We replace embeddings for new tokens and region tokens
                
                # 1. Inject New Token Embeddings (e.g. <r0>, <p>)
                if new_token_mask.any():
                    new_ids = input_ids - ori_vocab_size
                    new_ids = new_ids.masked_fill(~new_token_mask, 0)
                    new_embeds = self.new_input_embs(new_ids)
                    output[new_token_mask] = new_embeds[new_token_mask].to(output.dtype)
                
                # 2. Inject Region Features (at <region> token positions)
                if region_features_proj is not None and self.reg_token_id is not None:
                    reg_mask = input_ids == self.reg_token_id
                    # Safety check
                    if reg_mask.sum() == region_features_proj.shape[0]:
                        output.masked_scatter_(reg_mask.unsqueeze(-1), region_features_proj.to(output.dtype))
                
                # 3. Inject Region Features at <refer_feat> token positions (for referring tasks)
                # This is critical for action referring tasks where <refer_feat> tokens need visual features
                if region_features_proj is not None and self.refer_feat_token_id is not None:
                    ref_feat_mask = input_ids == self.refer_feat_token_id
                    # Safety check: number of <refer_feat> tokens should match number of region features
                    if ref_feat_mask.sum() > 0 and ref_feat_mask.sum() == region_features_proj.shape[0]:
                        output.masked_scatter_(ref_feat_mask.unsqueeze(-1), region_features_proj.to(output.dtype))
                    
                return output

            # Register Hook
            hook_handle = self.model.get_input_embeddings().register_forward_hook(embedding_hook)
            
            try:
                # 3. Pass to Qwen with masked_input_ids
                return super().forward(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=None, # We let Qwen compute embeddings (which triggers our hook)
                    labels=labels,
                    use_cache=False if self.training else use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    **kwargs
                )
            finally:
                # Remove hook
                hook_handle.remove()
        else:
            # Fallback if input_ids is None (e.g. generation loop with past_key_values?)
            # If input_ids is None, Qwen might crash anyway as seen before.
            # But if inputs_embeds IS provided (e.g. by user), we pass it.
            return super().forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
