import os
import json
import torch
from PIL import Image
from torchvision.ops import box_convert
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List

from groma.constants import DEFAULT_TOKENS, IGNORE_INDEX
from groma.data.conversation import conv_templates
from groma.data.datasets.det_data import normalize_box_coordinates


class GromaInstruct(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default'):
        super(GromaInstruct, self).__init__()
        self.meta_data = json.load(open(ann_file, "r"))
        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

    def __len__(self):
        return len(self.meta_data)

    def preprocess(self, data_item):
        bboxes = data_item['boxes']
        bboxes = torch.tensor(bboxes)
        bboxes = box_convert(bboxes, 'xywh', 'cxcywh')
        img_w = data_item['width']
        img_h = data_item['height']
        bboxes = normalize_box_coordinates(bboxes, (img_h, img_w))
        conversations = data_item['conversation']

        refer_boxes_list = []  # For boxes in human queries (referring task)
        ground_boxes_list = []  # For boxes in assistant responses (grounding task)
        new_conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))

        assert len(conversations) % 2 == 0
        for i, conversation in enumerate(conversations):
            chat = conversation['value']
            box_inds = conversation.get('box_inds', None)

            if i % 2 == 0:
                # Human query
                chat = DEFAULT_TOKENS['ground'] + chat
                # If human query has box_inds, these are referring boxes
                # IMPORTANT: Add ALL box references including duplicates (e.g., [0,0,1,1] -> add box 0 twice, box 1 twice)
                if box_inds is not None and len(box_inds) > 0:
                    refer_boxes_list.extend([bboxes[idx] for idx in box_inds])
            else:
                # Assistant response
                chat = DEFAULT_TOKENS['sep'] + chat + DEFAULT_TOKENS['sep']
                # If assistant response has box_inds, these are ground truth boxes
                if box_inds is not None and len(box_inds) > 0:
                    ground_boxes_list.extend([bboxes[idx] for idx in box_inds])

            new_conversations.append((self.conv_temp.roles[i%2], chat))

        prompt = self.conv_temp.get_prompt(new_conversations)

        # Stack boxes appropriately
        if len(ground_boxes_list) > 0:
            ground_boxes = torch.stack(ground_boxes_list)
        else:
            ground_boxes = torch.empty(0, 4)

        if len(refer_boxes_list) > 0:
            refer_boxes = torch.stack(refer_boxes_list)
        else:
            refer_boxes = torch.empty(0, 4)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids[0]

        # Mask targets
        targets = input_ids.clone()
        sep_inds = (input_ids == self.seperator_id).nonzero(as_tuple=True)[0]

        # Handle truncation cases: if truncation caused odd number of <sep> tokens,
        # remove the last unpaired <sep> token
        if len(sep_inds) % 2 != 0:
            # Truncation likely cut off the closing <sep>, remove the last unpaired one
            sep_inds = sep_inds[:-1]
            if len(sep_inds) == 0:
                # No valid sep pairs, this is a malformed sample - skip masking
                pass
        for i in range(0, len(sep_inds), 2):
            pre_sep = 0 if i == 0 else sep_inds[i - 1]
            cur_sep = sep_inds[i]
            targets[pre_sep:cur_sep] = IGNORE_INDEX
        eos_inds = (input_ids == self.eos_id).nonzero(as_tuple=True)[0]
        targets[eos_inds[1:]] = self.eos_id

        # Remove sep token
        mask = input_ids != self.seperator_id
        input_ids = input_ids[mask]
        targets = targets[mask]

        data_dict = dict(
            input_ids=input_ids,
            labels=targets,
            ground_boxes=ground_boxes,
            refer_boxes=refer_boxes,
            source='walle_data'
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.meta_data[i]
        image_file = data_item['file_name']
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB').resize((448, 448))
        image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict = self.preprocess(data_item)
        data_dict['image'] = image
        return data_dict
