"""
[GROMA-QWEN V3] Groma Qwen3VL with Interaction Token Architecture

This module extends Groma Qwen V2 Native by adding an interaction token that
represents the union bounding box between person and object regions.

Architecture Overview:
======================
GromaQwenInteractionModel extends GromaQwenModel (V2) to add a third region
feature representing the spatial context of the human-object interaction.

Key Changes from V2:
--------------------
1. Interaction Box: Automatically computed as union(person_box, object_box)
2. Three Region Features: person, object, and interaction (instead of 2)
3. Same External API: User provides only person + object boxes
4. Internal Augmentation: Interaction box added to prompt internally

Data Flow:
----------
Input: Image + 2 bounding boxes (person, object)
1. Compute interaction_box = union(person_box, object_box)
2. Internally create prompt with 3 boxes: person, object, interaction
3. Full Image -> Qwen3VL Vision Encoder -> Features from layers [8, 16, 24]
4. 3 Bounding boxes -> ROI Align -> 3 Region features (4096-dim each)
5. Region features -> Image-to-Text Bridge -> Projected features
6. Embeddings injected at 3 <|box_end|> positions via forward hook
7. LLM generates action description

Components (same as V2, shared):
--------------------------------
1. Base LLM: Qwen3VL (frozen during training)
2. Base Vision: Qwen3VL Vision Encoder (frozen, reused for region features)
3. Region Encoder (region_encoder): MLVLROIQueryModule - shared for all 3 boxes
4. Image-to-Text Bridge (img_txt_bridge): Projects region features to LLM space

Training Strategy:
------------------
Stage 2 Only (frozen Qwen3VL):
- Freeze: Qwen3VL LLM + Qwen3VL Vision Encoder
- Train: ROI Align module + Image-to-Text Bridge (learns interaction context)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from groma.model.groma_qwen import (
    GromaQwenModel,
    GromaQwenConfig,
    QWEN3VL_BOX_START_TOKEN_ID,
    QWEN3VL_BOX_END_TOKEN_ID,
)


def compute_union_box(boxes: torch.Tensor) -> torch.Tensor:
    """Compute the union (bounding box) of multiple boxes.
    
    Args:
        boxes: Tensor of shape [N, 4] with boxes in [x1, y1, x2, y2] format
               Coordinates should be in [0, 1000] range (Qwen3VL format)
    
    Returns:
        Union box of shape [1, 4] in [x1, y1, x2, y2] format
    """
    if boxes.shape[0] == 0:
        return torch.zeros(1, 4, dtype=boxes.dtype, device=boxes.device)
    
    # Compute union: min of x1,y1 and max of x2,y2
    x1 = boxes[:, 0].min()
    y1 = boxes[:, 1].min()
    x2 = boxes[:, 2].max()
    y2 = boxes[:, 3].max()
    
    return torch.tensor([[x1, y1, x2, y2]], dtype=boxes.dtype, device=boxes.device)


def compute_interaction_box(person_box: torch.Tensor, object_box: torch.Tensor) -> torch.Tensor:
    """Compute the interaction box as the union of person and object boxes.
    
    Args:
        person_box: Tensor of shape [1, 4] or [4] with person box
        object_box: Tensor of shape [1, 4] or [4] with object box
    
    Returns:
        Interaction box of shape [1, 4]
    """
    # Ensure 2D tensors
    if person_box.dim() == 1:
        person_box = person_box.unsqueeze(0)
    if object_box.dim() == 1:
        object_box = object_box.unsqueeze(0)
    
    # Stack and compute union
    boxes = torch.cat([person_box, object_box], dim=0)
    return compute_union_box(boxes)


class GromaQwenInteractionConfig(GromaQwenConfig):
    """Configuration for GromaQwenInteractionModel.
    
    Extends GromaQwenConfig with interaction token settings.
    """
    model_type = "groma_qwen_interaction"

    def __init__(
        self,
        use_interaction_token: bool = True,
        interaction_position: str = "after_object",  # "after_object" or "after_person"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_interaction_token = use_interaction_token
        self.interaction_position = interaction_position


class GromaQwenInteractionModel(GromaQwenModel):
    """Groma-Qwen Model with Interaction Token.
    
    This model extends GromaQwenModel (V2) to support an additional interaction
    region that represents the union of person and object bounding boxes.
    
    The interaction token provides spatial context for understanding
    human-object interactions by capturing the joint region where the
    interaction occurs.
    
    Key Features:
    - Automatic interaction box computation from person + object boxes
    - Same external API as V2 (backward compatible)
    - Three ROI features: person, object, interaction
    - Shared ROI encoder and bridge for all regions
    """
    config_class = GromaQwenInteractionConfig

    def __init__(self, config: GromaQwenInteractionConfig):
        # Initialize parent (GromaQwenModel V2)
        super().__init__(config)
        
        # Interaction token settings
        self.use_interaction_token = getattr(config, 'use_interaction_token', True)
        self.interaction_position = getattr(config, 'interaction_position', 'after_object')
        
        print(f"[GromaQwenInteractionModel V3] Initialized with:")
        print(f"  - Use interaction token: {self.use_interaction_token}")
        print(f"  - Interaction position: {self.interaction_position}")
        print(f"  - Sharing ROI encoder and bridge with person/object tokens")

    def _augment_boxes_with_interaction(
        self,
        refer_boxes: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Augment refer_boxes with interaction boxes.
        
        For each batch item, computes the union of person and object boxes
        and appends the interaction box.
        
        Args:
            refer_boxes: List of [N, 4] tensors, where N is typically 2
                        (person_box, object_box)
        
        Returns:
            Augmented list of [N+1, 4] tensors with interaction box appended
        """
        if not self.use_interaction_token:
            return refer_boxes
        
        augmented_boxes = []
        for boxes in refer_boxes:
            if boxes.shape[0] >= 2:
                # Compute interaction box from first two boxes (person, object)
                person_box = boxes[0:1]  # Keep as [1, 4]
                object_box = boxes[1:2]  # Keep as [1, 4]
                interaction_box = compute_interaction_box(person_box, object_box)
                
                # Append interaction box
                augmented = torch.cat([boxes, interaction_box.to(boxes.device)], dim=0)
                augmented_boxes.append(augmented)
            else:
                # Not enough boxes for interaction, keep as is
                augmented_boxes.append(boxes)
        
        return augmented_boxes

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
        refer_boxes: Optional[List[torch.Tensor]] = None,
        vision_hidden_states: Optional[List[torch.Tensor]] = None,
        # V3 specific - allow passing pre-augmented boxes
        include_interaction: Optional[bool] = None,
        **kwargs
    ):
        """Forward pass with interaction token support.
        
        Extends V2 forward by:
        1. Computing interaction boxes from person + object boxes
        2. Augmenting refer_boxes with interaction boxes
        3. Processing 3 regions instead of 2
        
        Note: The input_ids should already contain 3 <|box_end|> tokens
        (added by the dataset processor during training, or by generate() during inference)
        """
        
        # Determine whether to include interaction
        if include_interaction is None:
            include_interaction = self.use_interaction_token
        
        # Augment refer_boxes with interaction boxes if enabled
        if include_interaction and refer_boxes is not None:
            refer_boxes = self._augment_boxes_with_interaction(refer_boxes)
        
        # Call parent forward with augmented boxes
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
            refer_boxes=refer_boxes,
            vision_hidden_states=vision_hidden_states,
            **kwargs
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        refer_boxes: Optional[List[torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        include_interaction: Optional[bool] = None,
        **kwargs
    ):
        """Generate with interaction token support.
        
        Augments refer_boxes with interaction boxes before generation.
        The input_ids should already contain 3 <|box_end|> tokens.
        """
        # Determine whether to include interaction
        if include_interaction is None:
            include_interaction = self.use_interaction_token
        
        # Augment refer_boxes with interaction boxes if enabled
        if include_interaction and refer_boxes is not None:
            refer_boxes = self._augment_boxes_with_interaction(refer_boxes)
        
        # Pre-extract vision features
        vision_hidden_states = None
        if refer_boxes is not None and pixel_values is not None:
            vision_hidden_states = self._extract_vision_features(pixel_values, image_grid_thw)
        
        return super(GromaQwenModel, self).generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            refer_boxes=refer_boxes,
            vision_hidden_states=vision_hidden_states,
            **kwargs
        )

    @classmethod
    def from_v2_checkpoint(cls, v2_checkpoint_path: str, **kwargs):
        """Load from a V2 checkpoint and convert to V3.
        
        This allows reusing trained V2 weights as initialization for V3.
        The ROI encoder and bridge weights are compatible since they're shared.
        
        Args:
            v2_checkpoint_path: Path to V2 checkpoint
            **kwargs: Additional config overrides
        
        Returns:
            GromaQwenInteractionModel initialized from V2 weights
        """
        from groma.model.groma_qwen import GromaQwenConfig, GromaQwenModel
        
        # Load V2 config using the custom config class (not AutoConfig)
        v2_config = GromaQwenConfig.from_pretrained(v2_checkpoint_path, trust_remote_code=True)
        
        # Create V3 config with V2 settings
        v3_config = GromaQwenInteractionConfig(
            use_interaction_token=kwargs.pop('use_interaction_token', True),
            interaction_position=kwargs.pop('interaction_position', 'after_object'),
            **v2_config.to_dict()
        )
        
        # Load V2 model first, then create V3 model with same weights
        print(f"   Loading V2 model weights from: {v2_checkpoint_path}")
        v2_model = GromaQwenModel.from_pretrained(
            v2_checkpoint_path,
            config=v2_config,
            trust_remote_code=True,
        )
        
        # Create V3 model with V3 config
        print(f"   Creating V3 model with interaction token support...")
        v3_model = cls(v3_config)
        
        # Copy weights from V2 to V3 (they share the same architecture)
        # V3 only adds interaction computation logic, no new weights
        v3_model.load_state_dict(v2_model.state_dict(), strict=True)
        
        # Clean up V2 model to free memory
        del v2_model
        
        print(f"   ✓ Successfully converted V2 checkpoint to V3")
        
        return v3_model


# Utility functions for prompt building with interaction token

def build_interaction_prompt(
    person_box: List[int],
    object_box: List[int],
    include_interaction: bool = True,
    object_label: str = "object",
) -> str:
    """Build a prompt with person, object, and optionally interaction boxes.
    
    Args:
        person_box: [x1, y1, x2, y2] in [0, 1000] range
        object_box: [x1, y1, x2, y2] in [0, 1000] range
        include_interaction: Whether to include interaction box
        object_label: Label for the object (e.g., "bicycle", "bench")
    
    Returns:
        Formatted prompt string with native Qwen3VL bbox tokens
    """
    # Format boxes
    person_str = f"<|object_ref_start|>person<|object_ref_end|><|box_start|>({person_box[0]},{person_box[1]}),({person_box[2]},{person_box[3]})<|box_end|>"
    object_str = f"<|object_ref_start|>{object_label}<|object_ref_end|><|box_start|>({object_box[0]},{object_box[1]}),({object_box[2]},{object_box[3]})<|box_end|>"
    
    if include_interaction:
        # Compute interaction box
        interaction_box = [
            min(person_box[0], object_box[0]),
            min(person_box[1], object_box[1]),
            max(person_box[2], object_box[2]),
            max(person_box[3], object_box[3]),
        ]
        interaction_str = f"<|object_ref_start|>interaction<|object_ref_end|><|box_start|>({interaction_box[0]},{interaction_box[1]}),({interaction_box[2]},{interaction_box[3]})<|box_end|>"
        
        prompt = f"Action Recognition Task: The first region {person_str} contains a PERSON. The second region {object_str} contains an OBJECT. The interaction region {interaction_str} shows where the interaction occurs. Describe the action the person is performing with this object. Respond with only the action phrase."
    else:
        prompt = f"Action Recognition Task: The first region {person_str} contains a PERSON. The second region {object_str} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase."
    
    return prompt


def format_box_for_qwen(box: List[int]) -> str:
    """Format a box for Qwen3VL native format.
    
    Args:
        box: [x1, y1, x2, y2] in [0, 1000] range
    
    Returns:
        Formatted string like "(x1,y1),(x2,y2)"
    """
    return f"({box[0]},{box[1]}),({box[2]},{box[3]})"

