import collections
import numpy as np
import json
import os
import pickle
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS


class HICOEvaluator(object):
    ''' Evaluator for HICO-DET dataset '''
    def __init__(self, anno_file, output_dir, evaluation_mode='default'):
        """
        Initialize HICO evaluator with paper-standard protocol

        Args:
            anno_file: Path to annotation file
            output_dir: Output directory for results
            evaluation_mode: 'default' or 'known_object'
                - 'default': Evaluate on all test images (harder, includes background rejection)
                - 'known_object': Evaluate only on images containing target object (easier, focuses on interaction)
        """
        size = 600
        self.size = size
        self.anno_file = anno_file
        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.hico_ap  = np.zeros(size)
        self.hico_rec = np.zeros(size)
        self.output_dir = output_dir
        self.evaluation_mode = evaluation_mode

        # Build image-object index for known_object mode
        self.object_to_images = {}  # {object_coco_id: [list of image_ids]}
        self.image_to_objects = {}  # {image_id: [list of object_coco_ids]}
        self.hoi_to_object_id = {}  # {hoi_id: object_coco_id}

        if evaluation_mode == 'known_object':
            print(f"🔍 Building image-object index for Known Object evaluation mode...")
            self.object_to_images, self.image_to_objects = self.build_image_object_index(anno_file)
            self.hoi_to_object_id = self.build_hoi_to_object_mapping()
            print(f"✅ Image-object index built: {len(self.image_to_objects)} images, {len(self.object_to_images)} object types")

    def get_rare_non_rare_categories_paper_standard(self):
        """
        Follow CVPR 2024 paper specification:
        - 138 Rare interactions (< 10 training instances)
        - 462 Non-Rare interactions (≥ 10 training instances)
        - 600 Total interactions (includes no_interaction categories)

        Returns:
            tuple: (rare_ids, non_rare_ids, valid_ids)
        """
        from .hico_categories import RARE_INTERACTION_IDS

        rare_ids = set(RARE_INTERACTION_IDS)      # 138 categories
        all_ids = set(range(600))                 # All 600 categories
        non_rare_ids = all_ids - rare_ids         # 462 categories

        return rare_ids, non_rare_ids, all_ids

    def build_image_object_index(self, anno_file):
        """
        Build mappings between images and objects for Known Object evaluation.

        This creates an index to determine which test images contain which objects.
        Used to filter the test set per HOI category in Known Object mode.

        Returns:
            object_to_images: {object_coco_id: [list of image_ids with that object]}
            image_to_objects: {image_id: [list of object_coco_ids in that image]}
        """
        with open(anno_file, 'r') as f:
            dataset_dicts = json.load(f)

        image_to_objects = {}
        object_to_images = collections.defaultdict(list)

        for anno_dict in dataset_dicts:
            image_id = anno_dict['img_id']
            box_annos = anno_dict.get('annotations', [])

            # Extract all object categories in this image
            objects_in_image = []
            for box_anno in box_annos:
                obj_category_id = box_anno['category_id']
                objects_in_image.append(obj_category_id)

            # Store mapping
            image_to_objects[image_id] = list(set(objects_in_image))  # Remove duplicates

            # Build reverse mapping
            for obj_id in set(objects_in_image):
                object_to_images[obj_id].append(image_id)

        # Convert to regular dict and remove duplicates in lists
        object_to_images = {k: list(set(v)) for k, v in object_to_images.items()}

        return object_to_images, image_to_objects

    def build_hoi_to_object_mapping(self):
        """
        Build mapping from HOI ID to object COCO ID.

        Returns:
            hoi_to_object_id: {hoi_id: object_coco_id}
        """
        # Create object name to COCO ID mapping
        object_name_to_id = {obj['name']: obj['id'] for obj in HICO_OBJECTS}

        # Map each HOI to its target object COCO ID
        hoi_to_object_id = {}
        for interaction in HICO_INTERACTIONS:
            hoi_id = interaction['interaction_id']
            object_name = interaction['object']
            object_coco_id = object_name_to_id.get(object_name)

            if object_coco_id is not None:
                hoi_to_object_id[hoi_id] = object_coco_id
            else:
                print(f"Warning: Object '{object_name}' not found in HICO_OBJECTS mapping")

        return hoi_to_object_id

    def update(self, predictions):
        ''' Store predictions
        Args:
            predictions (dict): a dictionary in the following format.
            {
                img_id: [
                    [hoi_id, score, pbox_x1, pbox_y1, pbox_x2, pbox_y2, obox_x1, obox_y1, obox_x2, obox_y2],
                    ...
                    ...
                ]
            }
        '''
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)

    def accumulate(self):
        """Calculate AP for all HOI categories using the specified evaluation mode"""
        if self.evaluation_mode == 'known_object':
            # Known Object mode: Only evaluate on images that contain the target object
            # This filters the test set per HOI category (easier task, focuses on interaction)
            for hoi_id in range(600):
                gts_per_hoi = self.gts[hoi_id]

                # Get target object COCO ID for this HOI
                target_object_id = self.hoi_to_object_id.get(hoi_id)

                if target_object_id is None:
                    # If no object mapping (shouldn't happen), fall back to default
                    ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
                else:
                    # Get images containing this object (filtered test set)
                    valid_image_ids = set(self.object_to_images.get(target_object_id, []))

                    # Calculate AP only on images containing the target object
                    ap, rec = calc_ap_known_object(
                        self.scores[hoi_id],
                        self.boxes[hoi_id],
                        self.keys[hoi_id],
                        gts_per_hoi,
                        valid_image_ids
                    )

                self.hico_ap[hoi_id], self.hico_rec[hoi_id] = ap, rec
        else:
            # Default mode: Evaluate on ALL test images (harder task, includes background rejection)
            for hoi_id in range(600):
                gts_per_hoi = self.gts[hoi_id]
                ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
                self.hico_ap[hoi_id], self.hico_rec[hoi_id] = ap, rec

    def summarize(self, ckpt_num=""):
        """
        Calculate metrics following paper-standard protocol:
        138 Rare + 462 Non-Rare = 600 Total
        """
        rare_ids, non_rare_ids, valid_ids = self.get_rare_non_rare_categories_paper_standard()

        # Convert to numpy arrays for indexing
        rare_hois = np.array(list(rare_ids))
        non_rare_hois = np.array(list(non_rare_ids))
        valid_hois = np.array(list(valid_ids))

        # Calculate mAP for each category
        rare_mAP = np.mean(self.hico_ap[rare_hois])
        non_rare_mAP = np.mean(self.hico_ap[non_rare_hois])
        full_mAP = np.mean(self.hico_ap[valid_hois])

        # Print results
        print(f"{self.evaluation_mode.upper()} SETTING (Paper-Standard Protocol):")
        print("Full mAP: {:.4f} ({:.2f}%) - 600 categories".format(full_mAP, full_mAP * 100.))
        print("Rare mAP: {:.4f} ({:.2f}%) - 138 categories".format(rare_mAP, rare_mAP * 100.))
        print("Non-Rare mAP: {:.4f} ({:.2f}%) - 462 categories".format(non_rare_mAP, non_rare_mAP * 100.))

        # Store metrics for JSON output
        self.last_metrics = {
            "evaluation_mode": self.evaluation_mode,
            "protocol": "paper_standard",
            self.evaluation_mode: {
                "full_mAP": float(full_mAP),
                "rare_mAP": float(rare_mAP),
                "non_rare_mAP": float(non_rare_mAP),
                "full_mAP_percent": float(full_mAP * 100.),
                "rare_mAP_percent": float(rare_mAP * 100.),
                "non_rare_mAP_percent": float(non_rare_mAP * 100.),
                "rare_count": 138,
                "non_rare_count": 462,
                "total_count": 600
            },
            "dataset_type": "HICO-DET"
        }

    def save_preds(self):
        with open(os.path.join(self.output_dir, "preds.pkl"), "wb") as f:
            pickle.dump({"scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        with open(os.path.join(output_dir, "dets.pkl"), "wb") as f:
            pickle.dump({"gts": self.gts, "scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def load_anno(self, anno_file):
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)

        action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
        object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
        hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

        size = self.size
        gts = {i: collections.defaultdict(list) for i in range(size)}
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("annotations", [])
            hoi_annos = anno_dict.get("hoi_annotation", [])
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["category_id"] - 1 # original annotations start from 1
                object_id = box_annos[hoi["object_id"]]["category_id"] # original annotations start from 1
                hoi_id = hoi_mapper[(action_id2name[action_id], object_id2name[object_id])]
                gts[hoi_id][image_id].append(person_box + object_box)

        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

        return gts


def calc_ap_known_object(scores, boxes, keys, gt_boxes, valid_image_ids):
    """
    Calculate AP for Known Object setting.

    This mode filters the test set to only images containing the target object.
    For example, for "ride bicycle", only evaluate on images that have bicycles.
    This is an easier setting that focuses on interaction discrimination rather than
    object detection + interaction recognition.

    Args:
        scores: Prediction confidence scores
        boxes: Predicted bounding boxes (person + object)
        keys: Image IDs for predictions
        gt_boxes: Ground truth boxes per image
        valid_image_ids: Set of image IDs that contain the target object
                        (e.g., images with bicycles for bicycle-related HOIs)

    Returns:
        ap: Average Precision on the filtered test set
        rec: Maximum Recall on the filtered test set
    """
    if len(keys) == 0:
        return 0, 0

    # Convert to numpy arrays if needed
    if isinstance(boxes, list):
        scores, boxes, keys = np.array(scores), np.array(boxes), np.array(keys)

    # Filter predictions to only valid images (images containing target object)
    valid_mask = np.array([k in valid_image_ids for k in keys])
    scores_filtered = scores[valid_mask]
    boxes_filtered = boxes[valid_mask]
    keys_filtered = keys[valid_mask]

    # Filter ground truth to only valid images
    gt_boxes_filtered = {k: v for k, v in gt_boxes.items() if k in valid_image_ids}

    # If no predictions or GT in valid images, return 0
    if len(keys_filtered) == 0:
        return 0, 0

    # Calculate AP using filtered data
    hit = []
    idx = np.argsort(scores_filtered)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes_filtered.keys():
        npos += gt_boxes_filtered[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes_filtered[pair_id, :]
        key = keys_filtered[pair_id]
        if key in gt_boxes_filtered:
            maxi = 0.0
            k = -1
            for j in range(gt_boxes_filtered[key].shape[0]):
                tmp = calc_hit(box, gt_boxes_filtered[key][j, :])
                if maxi < tmp:
                    maxi = tmp
                    k = j
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)

    if len(hit) == 0:
        return 0, 0

    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec) if len(rec) else 0


def calc_ap(scores, boxes, keys, gt_boxes):

    if len(keys) == 0:
        return 0, 0

    if isinstance(boxes, list):
        scores, boxes, key = np.array(scores), np.array(boxes), np.array(keys)

    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k    = -1
            for i in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec) if len(rec) else 0


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


''' deprecated, evaluator
def hico_evaluation(predictions, gts):
    images, results = [], []
    for img_key, ps in predictions.items():
        images.extend([img_key] * len(ps))
        results.extend(ps)

    hico_ap, hico_rec = np.zeros(600), np.zeros(600)

    scores = [[] for _ in range(600)]
    boxes = [[] for _ in range(600)]
    keys = [[] for _ in range(600)]

    for img_id, det in zip(images, results):
        hoi_id, person_box, object_box, score = int(det[0]), det[1], det[2], det[-1]
        scores[hoi_id].append(score)
        boxes[hoi_id].append([float(x) for x in person_box] + [float(x) for x in object_box])
        keys[hoi_id].append(img_id)

    for hoi_id in range(600):
        gts_per_hoi = gts[hoi_id]
        ap, rec = calc_ap(scores[hoi_id], boxes[hoi_id], keys[hoi_id], gts_per_hoi)
        hico_ap[hoi_id], hico_rec[hoi_id] = ap, rec

    return hico_ap, hico_rec


def prepare_hico_gts(anno_file):
    """
    Convert dataset to the format required by evaluator.
    """
    with open(anno_file, "r") as f:
        dataset_dicts = json.load(f)

    action_mapper = {x["name"]: x["id"]+1 for x in HICO_ACTIONS}
    object_mapper = {x["name"]: x["id"] for x in HICO_OBJECTS}
    hoi_mapper = {(action_mapper[x["action"]], object_mapper[x["object"]]): x["interaction_id"]
                  for x in HICO_INTERACTIONS}

    gts = {i: collections.defaultdict(list) for i in range(600)}
    for anno_dict in dataset_dicts:
        image_id = int(anno_dict["file_name"].split("_")[-1].split(".")[0])
        box_annos = anno_dict.get("annotations", [])
        hoi_annos = anno_dict.get("hoi_annotation", [])
        for hoi in hoi_annos:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            action_id = hoi["category_id"]
            object_id = box_annos[hoi["object_id"]]["category_id"]
            hoi_id = hoi_mapper[(action_id, object_id)]
            gts[hoi_id][image_id].append(person_box + object_box)

    for hoi_id in gts:
        for img_id in gts[hoi_id]:
            gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

    return gts
'''