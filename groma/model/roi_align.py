# [GROMA-QWEN V2] ROI Align module for Groma Qwen3VL Native Architecture
# Part of: Groma Qwen3VL Native Architecture (No DINOv2, No New Tokens)
#
# Architecture Overview:
# ======================
# This module extracts and fuses multi-level features from Qwen3VL Vision Encoder
# for specific regions defined by bounding boxes.
#
# Components:
# -----------
# 1. MLVLFuseModule: Multi-level feature fusion with channel shuffling
#    - Takes features from Qwen3VL vision layers [8, 16, 24] (deepstack_visual_indexes)
#    - Fuses information across levels via channel shuffling
#    - Adds 2D coordinate features for spatial awareness
#
# 2. MLVLROIQueryModule: Main interface called by GromaQwenModel
#    - Reshapes Qwen3VL sequence features (B, L, D) -> spatial (B, D, H, W)
#    - Upsamples features to different resolutions for multi-scale ROI
#    - Applies MLVLFuseModule then MlvlRoIExtractor
#
# 3. MlvlRoIExtractor: ROI Align with positional encoding
#    - Applies ROI Align at each feature level
#    - Adds positional embedding from bounding box coordinates
#    - Projects to 4096-dim output for LLM embedding space
#
# Key Dimensions (Qwen3VL):
# -------------------------
# - Qwen3VL Vision hidden_states dim: 4096 (post-projection, matches LLM)
#   Note: The raw vision encoder outputs 1152, but output_hidden_states=True
#   returns post-projection features at 4096-dim
# - deepstack_visual_indexes: [8, 16, 24] for multi-level features
# - Output dimension: 4096 (matches LLM hidden size)
#
# Data Flow:
# ----------
# Qwen3VL vision features (B, L, 4096) x 3 levels [from layers 8, 16, 24]
#   -> Reshape to (B, 4096, H, W)
#   -> Upsample to multi-scale for ROI extraction
#   -> MLVLFuseModule: channel shuffle + coordinate encoding
#   -> MlvlRoIExtractor: ROI Align (14x14 output) + position embedding
#   -> Output: (N_regions, 4096) features for Image-to-Text Bridge
#
import re
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Linear, normal_init
from mmdet.models import BaseRoIExtractor

# Qwen3VL Vision Encoder hidden size (post-projection from output_hidden_states)
# Note: Raw vision patch embeddings are 1152, but output_hidden_states returns
# post-projection features at 4096-dim to match LLM hidden size
QWEN3VL_VISION_HIDDEN_SIZE = 4096


def str2spi(input_str):
    bbox_regex = r'<bbox>\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*</bbox>'
    # only attention inter the instruction
    results = []
    matches = re.findall(bbox_regex, input_str)
    for match in matches:
        results.append([float(match[0]), float(match[1]), float(match[2]),
                        float(match[3])])
    return results


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def coordinate_to_encoding(coord_tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos


def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])


def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs


class MLVLFuseModule(nn.Module):
    """Multi-level feature fusion module.
    
    Args:
        input_dims: Input feature dimension (4096 for Qwen3VL post-projection)
        embed_dims: Embedding dimension for fusion (same as input_dims)
        num_levels: Number of feature levels (3 for both Qwen3VL and DINOv2)
        num_fuse: Number of fusion iterations
    """
    def __init__(self, input_dims=QWEN3VL_VISION_HIDDEN_SIZE, embed_dims=QWEN3VL_VISION_HIDDEN_SIZE, 
                 num_levels=3, num_fuse=4):
        super(MLVLFuseModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_fuse = num_fuse
        self.input_dims = input_dims
        self.shuffle_channles = embed_dims // 4

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = self.num_levels
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.embed_dims - self.shuffle_channles * 2
        self._init_layers()

    def generate_coordinate(self, featmap_sizes, device='cuda'):
        x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
        y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')
        y = y.expand([featmap_sizes[0], 1, -1, -1])
        x = x.expand([featmap_sizes[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        return coord_feat

    def _init_layers(self):
        self.input_conv = nn.ModuleList(
            [nn.Conv2d(self.input_dims + 2, self.embed_dims, 1) for _ in range(self.num_levels)])
        self.fuse_convs = nn.ModuleList()
        for i in range(self.num_fuse):
            self.fuse_convs.append(
                ConvModule(self.embed_dims,
                           self.embed_dims,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=None,
                           norm_cfg=dict(type='GN',
                                         num_groups=64,
                                         requires_grad=True)
                           ))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        # Get dtype from inputs to ensure consistency
        input_dtype = inputs[0].dtype
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:,
                           self.remain_chs:][:,
                           self.shuffle_channles:]
                # Interpolate in float32 for precision, then convert back to input dtype
                from_top = F.interpolate(from_top.to(torch.float32),
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True).to(dtype=input_dtype)
                from_down = down_input[:, self.remain_chs:][:, :self.
                shuffle_channles]
                # Interpolate in float32 for precision, then convert back to input dtype
                from_down = F.interpolate(from_down.to(torch.float32),
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True).to(dtype=input_dtype)
                fused_inputs.append(
                    torch.cat([remain, from_top, from_down], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def forward(self, inputs):
        feat_size = [item.shape for item in inputs]
        # Get dtype from inputs to ensure consistency
        input_dtype = inputs[0].dtype
        new_inputs = []
        for feat, single_feat_size in zip(inputs, feat_size):
            coord_feat = self.generate_coordinate(single_feat_size, device=inputs[0].device)
            # Ensure coord_feat matches input dtype (coord_feat is float32 by default)
            coord_feat = coord_feat.to(dtype=input_dtype)
            feat = torch.cat([feat, coord_feat], dim=1)
            new_inputs.append(feat)
        inputs = new_inputs

        inputs = [self.input_conv[lvl](item) for lvl, item in enumerate(inputs)]

        for conv_m in self.fuse_convs:
            inputs = self._single_shuffle(inputs, [conv_m])
        return inputs


class MLVLROIQueryModule(nn.Module):
    """Multi-level ROI Query Module for extracting region features.
    
    Args:
        embed_dims: Vision feature dimension (4096 for Qwen3VL post-projection)
        out_dims: Output dimension (4096 to match LLM hidden size)
        num_levels: Number of feature levels (3)
    """
    def __init__(self, embed_dims=QWEN3VL_VISION_HIDDEN_SIZE, out_dims=4096, num_levels=3):
        super(MLVLROIQueryModule, self).__init__()
        self.mlvl_fuse = MLVLFuseModule(
            input_dims=embed_dims,
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_fuse=5)
        strids = [14 / 8, 14 / 4, 14 / 2]
        assert len(strids) == num_levels
        bbox_roi_extractor = dict(
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=embed_dims,
            embed_dims=embed_dims,
            fuse_level=num_levels,
            featmap_strides=strids)

        self.roi_align = MlvlRoIExtractor(**bbox_roi_extractor)

    def forward(self, mlvl_feats, bboxes, image_grid_thw=None):
        """Forward pass for extracting region features.
        
        Args:
            mlvl_feats: List of multi-level features, each with shape [L, D] or [B, L, D] or [B, D, H, W]
            bboxes: List of bounding boxes for each image, normalized to [0, 1]
            image_grid_thw: Optional tensor with shape [B, 3] containing [T, H, W] grid dimensions
        """
        # Ensure dtype consistency - convert to match mlvl_fuse module dtype
        expected_dtype = next(self.mlvl_fuse.parameters()).dtype
        mlvl_feats = [f.to(dtype=expected_dtype) for f in mlvl_feats]
        
        # Handle 2D features [L, D] - add batch dimension
        if mlvl_feats[0].dim() == 2:
            mlvl_feats = [f.unsqueeze(0) for f in mlvl_feats]
        
        # Reshape from sequence to spatial format if needed
        if mlvl_feats[0].dim() == 3:
            b, seq_len, c = mlvl_feats[0].shape
            
            # Try to infer spatial dimensions from sequence length
            # For Qwen3VL with patch merging, find the best H, W
            if image_grid_thw is not None:
                # Use provided grid dimensions (typically half of original due to patch merging)
                t, h, w = image_grid_thw[0].tolist()
                # Qwen3VL may have different spatial reduction, try to find matching dims
                if h * w != seq_len:
                    # Try halved dimensions (common for 2x2 patch merging)
                    h, w = h // 2, w // 2
            else:
                h, w = None, None
            
            # If still no match, find factors closest to square
            if h is None or h * w != seq_len:
                # Find best rectangle from sequence length
                best_h, best_w = 1, seq_len
                for test_h in range(int(math.sqrt(seq_len)), 0, -1):
                    if seq_len % test_h == 0:
                        best_h = test_h
                        best_w = seq_len // test_h
                        break
                h, w = best_h, best_w
            
            # Reshape to spatial format [B, C, H, W]
            if h * w == seq_len:
                mlvl_feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in mlvl_feats]
            else:
                # This shouldn't happen if we found valid factors
                raise ValueError(f"Cannot reshape sequence length {seq_len} to spatial dimensions")
        
        # Interpolate all levels to standard sizes for multi-scale ROI
        # Base size should be 32x32 for MlvlRoIExtractor compatibility
        target_base_size = 32
        num_level = len(mlvl_feats)
        # Create multi-scale targets: 32x32, 64x64, 128x128 for levels 0, 1, 2
        to_shape = [(target_base_size * 2 ** level, target_base_size * 2 ** level) 
                    for level in range(num_level)]
        to_shape = to_shape[::-1]  # Reverse so coarsest level gets largest size
        
        for level in range(num_level):
            feat = mlvl_feats[level]
            shape = to_shape[level]
            mlvl_feats[level] = F.interpolate(feat, size=shape, mode='bilinear', align_corners=True)
        
        # Ensure dtype consistency after interpolation (interpolate may change dtype)
        expected_dtype = next(self.mlvl_fuse.parameters()).dtype
        mlvl_feats = [f.to(dtype=expected_dtype) for f in mlvl_feats]
        
        mlvl_feats = self.mlvl_fuse(mlvl_feats)

        return self.roi_align(mlvl_feats, bboxes)


class MlvlRoIExtractor(BaseRoIExtractor):
    """Multi-level ROI Extractor with positional encoding.
    
    Args:
        embed_dims: Vision feature dimension (4096 for Qwen3VL post-projection)
        
    Note: Internal dimension for positional embedding and fusion is 4096
    to match the vision feature dimension, then projects to 4096.
    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 embed_dims=QWEN3VL_VISION_HIDDEN_SIZE,
                 stride=1,
                 norm_init=True,
                 fuse_level=3,
                 finest_scale=56,
                 init_cfg=None):
        super(MlvlRoIExtractor, self).__init__(roi_layer, out_channels,
                                               featmap_strides, init_cfg)
        self.embed_dims = embed_dims
        self.finest_scale = finest_scale
        self.fuse_level = fuse_level
        self.norm_init = norm_init

        self.pconvs = nn.ModuleList(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1)
            for _ in range(self.fuse_level))
        
        # Position embedding: bbox (4) -> 256 -> embed_dims (4096)
        self.pos_embedd = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        
        # Output projection (identity when embed_dims = 4096, kept for compatibility)
        self.updims = nn.Linear(embed_dims, 4096)

        # Flatten ROI features: embed_dims * 14^2 -> embed_dims
        self.flatten_linear = nn.Linear(self.embed_dims * self.roi_layers[0].output_size[0] ** 2, embed_dims)

        self.norm_init_weights()

    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        # Get expected dtype from model parameters
        expected_dtype = next(self.pos_embedd.parameters()).dtype
        
        num_imgs = len(rois)
        batched_rois = torch.cat(rois, dim=0)
        # Convert rois to match pos_embedd dtype (bfloat16)
        batched_rois = batched_rois.to(dtype=expected_dtype)
        pos_embedd = self.pos_embedd(batched_rois)
        # Ensure feats are in the correct dtype
        feats = [f.to(dtype=expected_dtype) for f in feats]
        
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        if feats[0].dim() == 3:
            h = w = int(math.sqrt(feats[0].shape[1]))
            assert h == 32
            assert w == 32
            b, c = feats[0].shape[0], feats[0].shape[-1]
            feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in feats]
        new_rois = []
        for img_id, single_img_roi in enumerate(rois):
            # rescale to original img scale
            single_img_roi = single_img_roi.to(dtype=expected_dtype) * 448
            roi_img_id = single_img_roi.new_ones(len(single_img_roi), dtype=expected_dtype) * img_id
            single_img_roi = torch.cat([roi_img_id[:, None], single_img_roi], dim=1)
            new_rois.append(single_img_roi)
        rois = torch.cat(new_rois)

        roi_feats = feats[0].new_zeros(self.fuse_level,
                                       rois.size(0), self.out_channels, *out_size)

        for i in range(num_levels):
            if len(rois) > 0:
                rois_ = rois
                # roi_layers need float32 for precision, but convert back to expected_dtype
                roi_feats_t = self.roi_layers[i](feats[i].to(torch.float32), rois_.to(torch.float32))
                roi_feats[i] = roi_feats_t.to(dtype=expected_dtype)

            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        fuse_roi_feats = []
        for i in range(self.fuse_level):
            fuse_roi_feats.append(self.pconvs[i](roi_feats[i]))

        fuse_roi_feats = sum(fuse_roi_feats)
        fuse_roi_feats = F.relu(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats.flatten(1, -1)
        fuse_roi_feats = self.flatten_linear(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats + pos_embedd
        fuse_roi_feats = self.updims(fuse_roi_feats)
        query_feats = []
        for i in range(num_imgs):
            mask = rois[:, 0] == i
            query_feats.append(fuse_roi_feats[mask])

        return query_feats
