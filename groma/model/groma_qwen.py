"""
[GROMA-QWEN V2] Groma Qwen3VL Native Architecture Model Implementation
Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)

Architecture Overview:
======================
GromaQwenModel extends Qwen3VLForConditionalGeneration to support region-level understanding
using ONLY native Qwen3VL components - no external vision encoder (DINOv2) or new tokens.

Key Design Principles:
----------------------
1. Reuse Qwen3VL Vision Encoder: Extract multi-level features from Qwen3VL's own vision encoder
   using deepstack_visual_indexes [8, 16, 24] for ROI Align
2. Native Tokens Only: Use Qwen3VL's built-in bbox tokens (<|box_start|>, <|box_end|>, etc.)
   instead of custom tokens (<region>, <refer_feat>, etc.)
3. Minimal Additions: Only add ROI Align module + Image-to-Text Bridge
4. Forward Hook: Inject region features at <|box_end|> token positions

Components:
-----------
1. Base LLM: Qwen3VL (frozen during training)
2. Base Vision: Qwen3VL Vision Encoder (frozen, reused for region features)
3. Region Encoder (region_encoder): MLVLROIQueryModule
   - Multi-level feature fusion from Qwen3VL vision layers [8, 16, 24]
   - ROI Align to extract features for specific bounding boxes
   - Outputs 4096-dim region features
4. Image-to-Text Bridge (img_txt_bridge): Projects region features to LLM embedding space
   - Linear(4096 -> 4096) + GELU + Linear(4096 -> 4096)
   - Trained during Stage 2

Native Qwen3VL Tokens Used:
---------------------------
- <|object_ref_start|> (151646): Start of object reference
- <|object_ref_end|> (151647): End of object reference  
- <|box_start|> (151648): Start of bounding box
- <|box_end|> (151649): End of bounding box - INJECTION POINT for region features
- <|vision_start|> (151652): Start of vision content
- <|vision_end|> (151653): End of vision content

Data Flow (Referring Task):
---------------------------
Input: Image + 2 bounding boxes (person, object) + prompt with native bbox tokens
1. Full Image -> Qwen3VL Vision Encoder -> Extract features from layers [8, 16, 24]
2. Bounding boxes -> ROI Align on Qwen3VL features -> Region features (4096-dim per region)
3. Region features -> Image-to-Text Bridge -> Projected features (4096-dim)
4. Embeddings injected at <|box_end|> positions via forward hook
5. LLM generates action description (e.g., "riding bicycle")

Training Strategy:
------------------
Stage 2 Only (frozen Qwen3VL):
- Freeze: Qwen3VL LLM + Qwen3VL Vision Encoder
- Train: ROI Align module + Image-to-Text Bridge
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    PretrainedConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from groma.model.roi_align import MLVLROIQueryModule, QWEN3VL_VISION_HIDDEN_SIZE


# Native Qwen3VL bbox token IDs
QWEN3VL_BOX_START_TOKEN_ID = 151648
QWEN3VL_BOX_END_TOKEN_ID = 151649
QWEN3VL_OBJECT_REF_START_TOKEN_ID = 151646
QWEN3VL_OBJECT_REF_END_TOKEN_ID = 151647


class GromaQwenConfig(Qwen3VLConfig):
    """Configuration for GromaQwenModel.
    
    This config extends Qwen3VLConfig with Groma-specific settings.
    Note: num_new_token is kept for backward compatibility but set to 0
    since we use native Qwen3VL tokens only.
    """
    model_type = "groma_qwen"

    def __init__(
        self,
        roi_align_cfg=None,
        num_new_token=0,  # Kept for backward compatibility, always 0 in V2
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
        
        self.roi_align_cfg = roi_align_cfg if roi_align_cfg is not None else {}
        self.num_new_token = num_new_token  # Always 0 in V2, kept for compatibility
        
        # Explicitly set loss_type to suppress warning
        if not hasattr(self, "loss_type") or self.loss_type is None:
            self.loss_type = "ForCausalLMLoss"


class GromaQwenModel(Qwen3VLForConditionalGeneration):
    """Groma-Qwen Model with Native Qwen3VL Architecture.
    
    This model extends Qwen3VLForConditionalGeneration to support region-level
    understanding using only native Qwen3VL components:
    - Reuses Qwen3VL vision encoder for region features
    - Uses native bbox tokens (<|box_start|>, <|box_end|>)
    - Adds only ROI Align + Image-to-Text Bridge
    """
    config_class = GromaQwenConfig

    def __init__(self, config: GromaQwenConfig):
        super().__init__(config)
        
        # Get LLM hidden size
        text_embed_dim = getattr(config, "hidden_size", None)
        if text_embed_dim is None:
            if hasattr(config, "text_config"):
                text_embed_dim = getattr(config.text_config, "hidden_size", 4096)
            else:
                text_embed_dim = 4096  # Fallback default for 7B/8B models
        
        # Vision hidden states are post-projection, matching LLM hidden size (4096)
        # Note: config.vision_config.hidden_size = 1152 (raw patch embeddings)
        # but output_hidden_states returns 4096-dim post-projection features
        vision_hidden_size = QWEN3VL_VISION_HIDDEN_SIZE  # 4096 for post-projection features
        
        # Get deepstack_visual_indexes for multi-level features
        self.deepstack_visual_indexes = [8, 16, 24]  # Default Qwen3VL indexes
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "deepstack_visual_indexes"):
            self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        
        print(f"[GromaQwenModel V2] Initializing with:")
        print(f"  - Vision hidden size: {vision_hidden_size}")
        print(f"  - LLM hidden size: {text_embed_dim}")
        print(f"  - Deepstack visual indexes: {self.deepstack_visual_indexes}")
        print(f"  - Using native Qwen3VL bbox tokens (no new tokens)")
        
        # Region Encoder (ROI Align + Multi-Level Fusion) - TRAINED
        # Uses Qwen3VL vision hidden size (4096, post-projection)
        self.region_encoder = MLVLROIQueryModule(
            embed_dims=vision_hidden_size,
            out_dims=4096,
            num_levels=len(self.deepstack_visual_indexes)
        )
        
        # Image-to-Text Bridge - TRAINED
        # Projects region features to LLM space
        self.img_txt_bridge = nn.Sequential(
            nn.Linear(4096, text_embed_dim),
            nn.GELU(),
            nn.Linear(text_embed_dim, text_embed_dim),
        )
        
        # Native Qwen3VL bbox token IDs (will be verified in init_special_token_id)
        self.box_end_token_id = QWEN3VL_BOX_END_TOKEN_ID
        self.box_start_token_id = QWEN3VL_BOX_START_TOKEN_ID
        
        # Legacy attributes for backward compatibility (not used in V2)
        self.reg_token_id = None
        self.refer_feat_token_id = None
        self.box_idx_token_ids = None
        
        # Store vision hidden states from last forward pass
        self._cached_vision_hidden_states = None

    def init_special_token_id(self, tokenizer):
        """Initialize and verify native Qwen3VL bbox token IDs.
        
        Unlike V1, this method only verifies native tokens exist - no new tokens are added.
        """
        # Verify native bbox tokens
        self.box_end_token_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
        self.box_start_token_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
        
        if self.box_end_token_id == tokenizer.unk_token_id:
            raise ValueError(
                "Native Qwen3VL token <|box_end|> not found in tokenizer. "
                "Please ensure you're using Qwen3-VL tokenizer."
            )
        if self.box_start_token_id == tokenizer.unk_token_id:
            raise ValueError(
                "Native Qwen3VL token <|box_start|> not found in tokenizer. "
                "Please ensure you're using Qwen3-VL tokenizer."
            )
        
        print(f"[GromaQwenModel V2] Native bbox tokens verified:")
        print(f"  - <|box_start|>: {self.box_start_token_id}")
        print(f"  - <|box_end|>: {self.box_end_token_id}")
        
        # Legacy: Set to None as we don't use custom tokens
        self.reg_token_id = None
        self.refer_feat_token_id = None
        self.box_idx_token_ids = None

    def _extract_vision_features(self, pixel_values, image_grid_thw):
        """Extract multi-level features from Qwen3VL vision encoder.
        
        This method extracts hidden states from the vision encoder at the
        deepstack_visual_indexes layers for use in ROI Align.
        
        Args:
            pixel_values: Input image tensor
            image_grid_thw: Image grid dimensions (temporal, height, width)
        
        Returns:
            List of hidden states from deepstack layers (typically 3 tensors)
        """
        if pixel_values is None:
            return None
        
        # Get the visual model (vision encoder)
        visual = self.visual
        
        # Process through vision encoder with hidden states output
        # Qwen3VL returns: (final_hidden_state, [hidden_states_list])
        # where hidden_states_list contains tensors from deepstack_visual_indexes layers
        with torch.no_grad():
            vision_outputs = visual(
                pixel_values,
                grid_thw=image_grid_thw,
                output_hidden_states=True
            )
            
            # Qwen3VL vision encoder returns tuple: (last_hidden_state, hidden_states_list)
            # hidden_states_list is already extracted from deepstack layers [8, 16, 24]
            if isinstance(vision_outputs, tuple) and len(vision_outputs) > 1:
                hidden_states = vision_outputs[1]
                
                # hidden_states is a list of tensors, one per deepstack layer
                if isinstance(hidden_states, (list, tuple)):
                    # Use the hidden states directly - they're already from the right layers
                    mlvl_feats = list(hidden_states)
                else:
                    # Single tensor case
                    mlvl_feats = [hidden_states]
            elif hasattr(vision_outputs, 'hidden_states'):
                hidden_states = vision_outputs.hidden_states
                mlvl_feats = list(hidden_states) if isinstance(hidden_states, (list, tuple)) else [hidden_states]
            else:
                # Fallback: use the output directly
                if isinstance(vision_outputs, tuple):
                    mlvl_feats = [vision_outputs[0]]
                else:
                    mlvl_feats = [vision_outputs]
        
        return mlvl_feats

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
        refer_boxes=None,
        vision_hidden_states=None,  # Pre-extracted vision features
        **kwargs,
    ):
        """Override prepare_inputs_for_generation to preserve Groma parameters.
        
        Note: In V2, we pass vision_hidden_states instead of region_images,
        since we extract features from Qwen3VL vision encoder directly.
        """
        # Call parent's prepare_inputs_for_generation
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
        if refer_boxes is not None:
            model_inputs["refer_boxes"] = refer_boxes
        if vision_hidden_states is not None:
            model_inputs["vision_hidden_states"] = vision_hidden_states
        
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
        refer_boxes: Optional[List[torch.Tensor]] = None,  # Boxes for ROI Align (list of [N, 4] tensors)
        vision_hidden_states: Optional[List[torch.Tensor]] = None,  # Pre-extracted features
        **kwargs
    ):
        """Forward pass with region feature injection at <|box_end|> positions.
        
        Key differences from V1:
        - No DINOv2 encoder - uses Qwen3VL vision features
        - No new token embeddings - uses native Qwen3VL tokens
        - Injects at <|box_end|> positions instead of <region>/<refer_feat>
        """
        
        region_features_proj = None
        
        # 1. Check if we have regions to process
        has_regions = (
            refer_boxes is not None
            and (isinstance(refer_boxes, list) and len(refer_boxes) > 0 and refer_boxes[0].shape[0] > 0)
        )
        
        if has_regions:
            # 2. Get vision features (extract or use cached)
            if vision_hidden_states is not None:
                # Use pre-extracted features
                mlvl_feats = vision_hidden_states
            elif pixel_values is not None:
                # Extract features from Qwen3VL vision encoder
                mlvl_feats = self._extract_vision_features(pixel_values, image_grid_thw)
            else:
                # No vision input, skip region processing
                mlvl_feats = None
            
            if mlvl_feats is not None:
                # Get model dtype for consistency
                model_dtype = next(self.region_encoder.parameters()).dtype
                
                # Convert features to model dtype
                mlvl_feats = [f.to(dtype=model_dtype) for f in mlvl_feats]
                
                # Coordinate Alignment (0-1000 -> 0-1)
                norm_refer_boxes = [box.float() / 1000.0 for box in refer_boxes]
                
                # Region Encoder (ROI Align + Multi-Level Fusion)
                # Pass image_grid_thw for proper spatial reshaping
                region_features = self.region_encoder(mlvl_feats, norm_refer_boxes, image_grid_thw)
                
                # Project to LLM dim via Image-to-Text Bridge
                region_features_flat = torch.cat(region_features, dim=0)
                region_features_proj = self.img_txt_bridge(region_features_flat)
        
        # 3. Set up embedding hook if we have region features to inject
        if input_ids is not None and region_features_proj is not None:
            # Define Hook to inject at <|box_end|> positions
            def embedding_hook(module, inputs, output):
                """Inject region features at <|box_end|> token positions."""
                # Find <|box_end|> positions
                box_end_mask = input_ids == self.box_end_token_id
                
                # Safety check: number of <|box_end|> tokens should match region features
                num_box_end = box_end_mask.sum().item()
                num_features = region_features_proj.shape[0]
                
                if num_box_end > 0 and num_box_end == num_features:
                    # Inject region features at <|box_end|> positions
                    output.masked_scatter_(
                        box_end_mask.unsqueeze(-1).expand_as(output),
                        region_features_proj.to(output.dtype)
                    )
                elif num_box_end > 0 and num_features > 0:
                    # Partial injection if counts don't match
                    # This can happen during generation when only some tokens are processed
                    pass
                
                return output
            
            # Register Hook
            hook_handle = self.model.get_input_embeddings().register_forward_hook(embedding_hook)
            
            try:
                # 4. Forward through Qwen3VL
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
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
            # No region features to inject - standard forward pass
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs
            )
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        refer_boxes: Optional[List[torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """Generate with region feature injection.
        
        Pre-extracts vision features before generation to avoid repeated computation.
        """
        # Pre-extract vision features if we have regions
        vision_hidden_states = None
        if refer_boxes is not None and pixel_values is not None:
            vision_hidden_states = self._extract_vision_features(pixel_values, image_grid_thw)
        
        return super().generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            refer_boxes=refer_boxes,
            vision_hidden_states=vision_hidden_states,
            **kwargs
        )
