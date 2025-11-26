import torch
import transformers
from dataclasses import dataclass

from groma.constants import IGNORE_INDEX


@dataclass
class DataCollatorForHybridDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # Collect keys
        input_ids = [instance.get('input_ids', None) for instance in instances]
        labels = [instance.get('labels', None) for instance in instances]
        
        # Image inputs
        # 'region_images' for DINOv2 (was 'image' in base class)
        region_images = [instance.get('region_images', instance.get('image', None)) for instance in instances]
        # 'pixel_values' for Qwen
        pixel_values = [instance.get('pixel_values', None) for instance in instances]
        # 'image_grid_thw' for Qwen3VL
        image_grid_thw = [instance.get('image_grid_thw', None) for instance in instances]
        
        refer_boxes = [instance.get('refer_boxes', torch.empty(0, 4)) for instance in instances]
        ground_boxes = [instance.get('ground_boxes', torch.empty(0, 4)) for instance in instances]

        # Concatenate all region_images from all samples
        # Each sample can have 0 or more regions, so we concatenate all regions
        # region_images from dataset: (N, C, H, W) where N can be 0 or more
        region_images_list = []
        for ri in region_images:
            if ri is not None and ri.numel() > 0:
                # ri shape: (N, C, H, W) where N >= 0
                if ri.shape[0] > 0:  # Has regions
                    region_images_list.append(ri)
        
        if len(region_images_list) > 0:
            # Concatenate all regions: (total_regions, C, H, W)
            region_images = torch.cat(region_images_list, dim=0)
        else:
            # No regions in entire batch
            region_images = None

        # Process pixel_values (Qwen inputs)
        # Qwen inputs might be variable size, so we might need to concatenate or pad?
        # Usually for Qwen-VL, pixel_values is a flattened tensor of patches.
        # If batching variable size images, we concatenate them and use image_grid_thw to split.
        if all([x is not None for x in pixel_values]):
            # Concatenate all pixel_values
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
            
        if all([x is not None for x in image_grid_thw]):
            image_grid_thw = torch.stack(image_grid_thw)
        else:
            image_grid_thw = None

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        # Truncate sequences to model's max position embeddings (2048 for Qwen3-VL)
        # This prevents OOM errors from sequences longer than model capacity
        # Keep the LAST max_seq_length tokens to preserve assistant responses (most important for training)
        max_seq_length = 2048
        if input_ids.size(1) > max_seq_length:
            input_ids = input_ids[:, -max_seq_length:]
            labels = labels[:, -max_seq_length:]
            
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            region_images=region_images, # Renamed from images
            pixel_values=pixel_values,   # New for Qwen
            image_grid_thw=image_grid_thw, # New for Qwen
            refer_boxes=refer_boxes,
            ground_boxes=ground_boxes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        return batch


@dataclass
class DataCollatorForDetDataset(object):
    def __call__(self, instances):
        meta_keys = ('image', 'class_labels', 'bboxes', 'source')
        images, class_labels, bboxes, sources = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        images = torch.stack(images)
        assert len(list(set(sources))) == 1, "data in the same batch should have the same data source."
        labels = [{'class_labels': label, "boxes": bbox, "source": source} for
                  label, bbox, source in zip(class_labels, bboxes, sources)]
        batch = dict(images=images, labels=labels)
        return batch


@dataclass
class DataCollatorForDetEvalDataset(object):
    def __call__(self, instances):
        meta_keys = ('image', 'ori_shape')
        images, ori_shapes = tuple([instance.get(key, None) for instance in instances] for key in meta_keys)
        images = torch.stack(images)
        ori_shapes = torch.stack([torch.tensor(x[:2]) for x in ori_shapes])
        batch = dict(images=images, ori_shapes=ori_shapes)
        return batch
