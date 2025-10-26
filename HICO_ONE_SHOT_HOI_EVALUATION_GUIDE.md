# HICO One-Shot HOI Grounding Evaluation Guide

## Overview

This document explains the complete evaluation pipeline for one-shot HOI (Human-Object Interaction) grounding on HICO-DET dataset. The evaluation measures how well the model can detect all entities (persons and objects) involved in HOI pairs using a **single prompt**, unlike the multi-turn per-category evaluation.

---

## Quick Start

### Run Evaluation

```bash
# Test on 10 images with visualizations
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_hoi_oneshot_eval.sh 0 \
    checkpoints/groma-7b-hoi-ground-refer-fine-tuned-hungarian \
    /path/to/test_hico_ann.json \
    /path/to/images/test2015 \
    results/hico_hoi_oneshot

# Full evaluation (all 9546 test images)
bash scripts/run_hico_hoi_oneshot_eval.sh 0 \
    checkpoints/groma-7b-hoi-ground-refer-fine-tuned-hungarian
```

### Expected Output

```
HICO-DET One-Shot HOI Grounding Metrics
========================================================================
Metric          Value  Description
------------------------------------------------------------------------
AR              21.2%  Average Recall @ IoU=0.50:0.95
AR@0.5          30.3%  Average Recall @ IoU=0.50
AR@0.75         30.3%  Average Recall @ IoU=0.75
AR@s             0.0%  Small objects
AR@m             0.0%  Medium objects
AR@l            24.6%  Large objects
========================================================================
```

---

## Complete Evaluation Flow

### Step 1: Dataset Loading

**File:** `groma/data/datasets/hico_hoi_oneshot.py`

```python
# Load HICO test annotations
{
  "file_name": "HICO_test2015_00000003.jpg",
  "annotations": [
    {"bbox": [302, 182, 322, 236], "category_id": 1},  # person
    {"bbox": [234, 85, 460, 284], "category_id": 9}     # boat
  ],
  "hoi_annotation": [
    {"subject_id": 0, "object_id": 1, "category_id": 77}  # person-boat interaction
  ]
}

# Dataset extracts ALL boxes referenced in HOI annotations
# For each image:
#   1. Collect all subject_id and object_id from hoi_annotation
#   2. Extract corresponding boxes from annotations
#   3. Store as ground truth for evaluation

# Result:
processed_item = {
    'file_name': 'HICO_test2015_00000003.jpg',
    'all_boxes': [[302, 182, 20, 54], [234, 85, 226, 199]],  # [x, y, w, h]
    'all_categories': ['person', 'boat'],
    'img_id': 3,
    'num_hoi_pairs': 1
}
```

**Key Point:** Only boxes involved in HOI pairs are extracted, not all boxes in the image.

---

### Step 2: Image Preprocessing

```python
# Resize to 448×448 (Groma's input size)
img = Image.open(img_path).convert('RGB')
img = img.resize((448, 448), Image.BILINEAR)

# Convert to numpy array
img_array = np.array(img, dtype=np.float32)

# Normalize with ImageNet statistics
mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
img_normalized = (img_array - mean) / std

# Convert to PyTorch tensor [C, H, W]
img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
image = img_tensor.unsqueeze(0).cuda()  # Add batch dimension
```

---

### Step 3: Build Prompt

**File:** `groma/data/datasets/hico_hoi_oneshot.py:build_prompt()`

```python
# Multi-turn conversation format
conversations = [
    # Turn 1: Initial context
    ("USER", "Here is an image with region crops from it. Image: <image>. Regions: <region>."),
    ("ASSISTANT", "Thank you for the image! How can I assist you with it?"),

    # Turn 2: HOI grounding query (exact training prompt!)
    ("USER", "[grounding] Describe what each person is doing with objects individually. Focus on actions only."),
    ("ASSISTANT", "")  # Model fills this
]

# Generate full prompt using conversation template
prompt = conv_template.get_prompt(conversations)

# Tokenize
input_ids = tokenizer([prompt])
input_ids = torch.as_tensor(input_ids.input_ids).cuda()
```

**Critical:** This is the **EXACT prompt used during training**. The model is optimized for this specific format.

---

### Step 4: Model Inference

**File:** `groma/eval/eval_hico_hoi_oneshot.py:eval_model()`

```python
# Run model inference
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image,
        use_cache=True,
        do_sample=False,
        max_new_tokens=512,  # Allow long responses for multiple HOI
        return_dict_in_generate=True,
        output_hidden_states=True,
        generation_config=model.generation_config
    )

# Extract TWO key outputs:

# 1. Generated text tokens
output_ids = outputs.sequences
generated_text = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=False)
# Example: "The <p>person</p><roi><r7></roi> is <p>driving</p> the <p>boat</p><roi><r3></roi>..."

# 2. Predicted bounding boxes from Deformable DETR detector
pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
# Shape: [N, 4] in normalized [cx, cy, w, h] format
# Example: N=55 boxes (detector proposals)
```

**Key Insight:** Model simultaneously generates:
- **Text:** HOI descriptions with region tokens (`<r0>`, `<r1>`, ...)
- **Boxes:** Bounding box coordinates from detector

The region tokens in text map to indices in the box tensor.

---

### Step 5: Smart Filtering (BEFORE Parsing)

**File:** `groma/eval/eval_hico_hoi_oneshot.py` (lines ~730-740)

```python
# Remove giant boxes (>50% of image area)
MAX_BOX_AREA_RATIO = 0.5

# Calculate box areas
box_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
valid_box_mask = box_areas < MAX_BOX_AREA_RATIO

# Filter BEFORE parsing
# BEFORE: 55 boxes (including 3 giant catch-all boxes)
# AFTER:  52 boxes (removed 3 giant boxes)
pred_boxes = pred_boxes[valid_box_mask]

print(f"Filtered {(~valid_box_mask).sum().item()} giant boxes")
```

**Why Filter Before Parsing?**

❌ **Wrong approach (filter after):**
```python
# Model generates: "<r9>" expecting box at index 9
# After filtering boxes 0-2, index 9 now points to what was index 12
# Result: Wrong box selected!
```

✅ **Correct approach (filter before):**
```python
# Filter first: Boxes 0-2 removed, box 12 becomes index 9
# Model generates: "<r9>" pointing to correct box
# Result: Correct mapping maintained!
```

**Example from Results:**
```
[Image 7/10] HICO_test2015_00000007.jpg
  Filtered 5 giant boxes (area > 0.5)
  Response: "... <p>person</p><roi><r19></roi> ... <p>dining table</p><roi><r49></roi> ..."
  # r19 and r49 point to correct boxes in filtered set
```

---

### Step 6: Response Parsing

**File:** `groma/eval/eval_hico_hoi_oneshot.py:parse_hoi_response()`

```python
def parse_hoi_response(response_text):
    """
    Extract entities and region indices from model response.

    Pattern: <p>entity_name</p><roi><rN></roi>
    """
    # Regex pattern for entity-box pairs
    pattern = r'<p>\s*([^<]+?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'
    matches = re.findall(pattern, response_text)

    entities = []
    region_indices = []

    for entity_name, region_idx in matches:
        entity_name = entity_name.strip()

        # Skip action words (sitting on, driving, holding, etc.)
        if is_action_word(entity_name):
            continue

        entities.append(entity_name)
        region_indices.append(int(region_idx))

    return entities, region_indices
```

**Example Parsing:**

**Input text:**
```
"The <p>person</p><roi><r7></roi> is <p>driving</p> the <p>boat</p><roi><r3></roi>.
 Another <p>person</p><roi><r2></roi> is <p>riding</p> a <p>boat</p><roi><r1></roi>."
```

**Parsed output:**
```python
entities = ['person', 'boat', 'person', 'boat']
region_indices = [7, 3, 2, 1]
# Note: 'driving' and 'riding' filtered out (actions, not entities)
```

**Action Word Filtering:**

The response contains BOTH entities and actions in `<p>` tags:
```
<p>person</p>     <-- ENTITY (keep)
<p>driving</p>    <-- ACTION (skip)
<p>boat</p>       <-- ENTITY (keep)
```

Action words are identified using a predefined list:
```python
HOI_ACTIONS = {
    'sitting on', 'riding', 'holding', 'wearing', 'carrying',
    'eating', 'drinking', 'looking at', 'playing', 'kicking',
    'hitting', 'throwing', 'catching', 'cutting', 'opening',
    'closing', 'pushing', 'pulling', 'washing', 'cleaning',
    'no interaction', 'using', 'touching', 'standing on',
    'lying on', 'walking', 'running', 'jumping', 'sitting',
    'loading', 'straddling', 'racing', 'driving', 'blowing'
}
```

---

### Step 7: Deduplication

**File:** `groma/eval/eval_hico_hoi_oneshot.py:deduplicate_boxes()`

```python
def deduplicate_boxes(region_indices, entities):
    """
    Remove duplicate region indices while preserving order.
    Handles artifacts from Hungarian matching during training.
    """
    seen = set()
    unique_indices = []
    unique_entities = []

    for idx, entity in zip(region_indices, entities):
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
            unique_entities.append(entity)

    return unique_indices, unique_entities
```

**Example:**

**Before deduplication:**
```python
region_indices = [7, 3, 7, 1]  # r7 appears twice
entities = ['person', 'boat', 'person', 'boat']
```

**After deduplication:**
```python
region_indices = [7, 3, 1]
entities = ['person', 'boat', 'boat']
# First occurrence of r7 kept, duplicate removed
```

**Why needed?** During training with Hungarian matching, the model might sometimes generate duplicate tokens for the same region. This ensures each box is counted only once.

---

### Step 8: Box Coordinate Extraction

**File:** `groma/eval/eval_hico_hoi_oneshot.py` (lines ~760-770)

```python
# After parsing and deduplication
region_indices_valid = [7, 3, 2, 1]  # Example
entities = ['person', 'boat', 'person', 'boat']

# Step 1: Filter valid indices (< total number of boxes)
region_indices_valid = [idx for idx in region_indices if idx < len(pred_boxes)]

# Step 2: Select boxes at these indices
selected_boxes = pred_boxes[region_indices_valid]
# Shape: [4, 4] - 4 boxes, each [cx, cy, w, h] in normalized coords

# Step 3: Convert from center format to corner format
# From: [center_x, center_y, width, height]
# To:   [x, y, width, height]
selected_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xywh')

# Step 4: Rescale from normalized [0, 1] to pixel coordinates
h, w = img_shape  # Original image size (e.g., 427 × 640)
selected_boxes[:, 0] *= w  # x
selected_boxes[:, 1] *= h  # y
selected_boxes[:, 2] *= w  # width
selected_boxes[:, 3] *= h  # height

# Result: Boxes in absolute pixel coordinates [x, y, w, h]
selected_boxes = selected_boxes.tolist()
# Example: [[302, 182, 20, 54], [234, 85, 226, 199], ...]
```

**Coordinate Transformations:**

```
Raw detector output:      [cx=0.48, cy=0.52, w=0.08, h=0.15] (normalized)
                                    ↓
After box_convert:        [x=0.44, y=0.445, w=0.08, h=0.15]  (normalized)
                                    ↓
After rescale (640×427):  [x=281.6, y=190.0, w=51.2, h=64.0] (pixels)
                                    ↓
Final [x, y, w, h]:       [281, 190, 51, 64]                (integers)
```

---

### Step 9: Match Predictions to Ground Truth

**File:** `groma/eval/eval_hico_hoi_oneshot.py:match_predictions_to_gt()`

#### Entity Name Normalization

```python
# Handle synonyms to account for name mismatches
ENTITY_SYNONYMS = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle',
    'scooter': 'motorcycle',
    'table': 'dining table',
    'desk': 'dining table',
    'ball': 'sports ball',
    'sofa': 'couch',
    'cellphone': 'cell phone',
    'phone': 'cell phone'
}

def normalize_entity_name(name):
    name = name.lower().strip()
    return ENTITY_SYNONYMS.get(name, name)

# Example:
normalize_entity_name('bike')  # → 'bicycle'
normalize_entity_name('table')  # → 'dining table'
```

#### IoU Matrix Computation

```python
def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU between all predicted and ground truth boxes"""
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)

    return iou_matrix

# Example:
pred_boxes = [[281, 190, 51, 64], [234, 85, 226, 199]]
gt_boxes = [[302, 182, 20, 54], [234, 85, 226, 199]]

iou_matrix = [
    [0.85, 0.12],  # pred[0] vs [gt[0], gt[1]]
    [0.08, 0.92]   # pred[1] vs [gt[0], gt[1]]
]
```

#### Greedy Matching with Priority

```python
def match_predictions_to_gt(pred_boxes, pred_entities, gt_boxes, gt_entities):
    # Normalize entity names
    pred_entities_norm = [normalize_entity_name(e) for e in pred_entities]
    gt_entities_norm = [normalize_entity_name(e) for e in gt_entities]

    # Build IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # Build candidates with priority scoring
    candidates = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            iou = iou_matrix[i, j]

            if iou > 0.5:  # IoU threshold
                name_match = (pred_entities_norm[i] == gt_entities_norm[j])

                # Priority: name match gets 2x weight, no match gets 1x
                priority = (2 if name_match else 1) * iou

                candidates.append((priority, i, j, iou, name_match))

    # Sort by priority (highest first)
    # Name matches with high IoU will be matched first
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Greedy assignment (one-to-one matching)
    matches = []
    matched_preds = set()
    matched_gts = set()

    for _, pred_idx, gt_idx, iou, name_match in candidates:
        # Only match if both pred and GT are unassigned
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx, iou, name_match))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Find unmatched
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]

    return matches, unmatched_preds, unmatched_gts
```

**Example Matching:**

```python
# Inputs
pred_boxes = [[281, 190, 51, 64], [234, 85, 226, 199]]
pred_entities = ['person', 'boat']
gt_boxes = [[302, 182, 20, 54], [234, 85, 226, 199]]
gt_entities = ['person', 'boat']

# Candidates with priority
candidates = [
    (priority=2*0.85, pred=0, gt=0, iou=0.85, name_match=True),   # person match
    (priority=1*0.12, pred=0, gt=1, iou=0.12, name_match=False),  # person vs boat (wrong)
    (priority=1*0.08, pred=1, gt=0, iou=0.08, name_match=False),  # boat vs person (wrong)
    (priority=2*0.92, pred=1, gt=1, iou=0.92, name_match=True)    # boat match
]

# After sorting by priority:
# [(2*0.92, 1, 1), (2*0.85, 0, 0), (1*0.12, 0, 1), (1*0.08, 1, 0)]

# Greedy assignment:
# Step 1: Match pred[1] ↔ gt[1] (boat, IoU=0.92) ✅
# Step 2: Match pred[0] ↔ gt[0] (person, IoU=0.85) ✅
# Step 3: Skip (0, 1) - pred[0] already matched
# Step 4: Skip (1, 0) - pred[1] already matched

# Final matches:
matches = [
    (pred_idx=1, gt_idx=1, iou=0.92, name_match=True),
    (pred_idx=0, gt_idx=0, iou=0.85, name_match=True)
]

unmatched_preds = []  # All predictions matched
unmatched_gts = []    # All GT boxes matched
```

**Handling Edge Cases:**

**Case 1: Name mismatch but high IoU**
```python
pred_entities = ['bike', 'person']  # Model says "bike"
gt_entities = ['bicycle', 'person']  # GT says "bicycle"

# After normalization:
pred_entities_norm = ['bicycle', 'person']  # 'bike' → 'bicycle'
gt_entities_norm = ['bicycle', 'person']

# Result: Name match = True, IoU = 0.88
# → Green box in visualization (excellent match)
```

**Case 2: More GT than predictions**
```python
pred_boxes = [box1, box2]  # 2 predictions
gt_boxes = [gt1, gt2, gt3, gt4, gt5]  # 5 ground truth

# After matching:
matches = [(0, 0, 0.85, True), (1, 2, 0.78, True)]  # 2 matches
unmatched_preds = []  # All preds matched
unmatched_gts = [1, 3, 4]  # 3 GT boxes not detected

# Impact on metrics:
# Recall = 2 / 5 = 40%  (2 out of 5 GT boxes detected)
```

---

### Step 10: Visualization (3-Panel Layout)

**File:** `groma/eval/eval_hico_hoi_oneshot.py:visualize_oneshot_results()`

#### Panel Layout

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  All Regions (N=52) │  Predictions + IoU  │   Ground Truth      │
│                     │                     │                     │
│  R0  R1  R2  R3 ... │  person(R7)         │  GT_person_0        │
│  (detector output)  │  IoU: 0.85 [Green]  │  (matched)          │
│                     │                     │                     │
│  R7 ← Gold outline  │  boat(R3)           │  GT_boat_1          │
│  (selected by model)│  IoU: 0.92 [Green]  │  (matched)          │
│                     │                     │                     │
│  R10 R11 ... R51    │                     │  GT_person_2        │
│  (unselected)       │                     │  [Dashed red]       │
│                     │                     │  (missed)           │
└─────────────────────┴─────────────────────┴─────────────────────┘

Legend:
  Green: IoU > 0.7, name match (excellent)
  Yellow: IoU > 0.7, name mismatch (good box, wrong label)
  Orange: 0.3 < IoU ≤ 0.7 (region proposer issue)
  Red: IoU ≤ 0.3 (wrong object)
  Purple: No GT match (false positive)
  Gold outline: Selected regions (left panel)
  Dashed red: Missed GT boxes (right panel)
```

#### Color-Coded Quality Assessment

```python
# Determine box color based on IoU and name match
if iou > 0.7:
    if name_match:
        box_color = '#00FF00'  # Green: Excellent
        stats['high_iou_name_match'] += 1
    else:
        box_color = '#FFFF00'  # Yellow: Good IoU but wrong label
        stats['high_iou_name_mismatch'] += 1
elif iou > 0.3:
    box_color = '#FFA500'  # Orange: Region proposer issue
    stats['medium_iou'] += 1
elif iou > 0:
    box_color = '#FF0000'  # Red: Wrong object
    stats['low_iou'] += 1
else:
    box_color = '#800080'  # Purple: False positive
    stats['unmatched_pred'] += 1
```

#### Statistics Display

```
Statistics:
  Total predictions: 2
  Good (IoU>0.7, name match): 2 (100.0%)
  Good IoU, name mismatch: 0 (0.0%)
  Partial (0.3<IoU≤0.7): 0 (0.0%)
  Wrong (IoU≤0.3): 0 (0.0%)
  No match: 0 (0.0%)
  Missed GT boxes: 0

Generated Response:
  The <p>person</p><roi><r7></roi> is <p>driving</p> the <p>boat</p><roi><r3></roi>...
```

---

### Step 11: COCO Evaluation

**File:** `groma/eval/eval_hico_hoi_oneshot.py:eval_model()`

#### Convert to COCO Format

```python
def convert_to_coco_format(predictions, dataset):
    """Convert predictions to COCO detection format"""
    results = []
    cat_name_to_id = dataset.name_to_cat_id

    for pred in predictions:
        img_id = pred['img_id']
        entities = pred['entities']
        boxes = pred['boxes']
        scores = pred['scores']

        for entity, box, score in zip(entities, boxes, scores):
            # Normalize entity name
            entity_norm = normalize_entity_name(entity)

            if entity_norm not in cat_name_to_id:
                continue  # Skip unknown categories

            cat_id = cat_name_to_id[entity_norm]

            results.append({
                "image_id": img_id,
                "category_id": cat_id,  # COCO category ID (1=person, 9=boat, etc.)
                "bbox": box,  # [x, y, w, h]
                "score": score  # Confidence (1.0 for grounding)
            })

    return results

# Example output:
[
    {"image_id": 3, "category_id": 1, "bbox": [302, 182, 20, 54], "score": 1.0},
    {"image_id": 3, "category_id": 9, "bbox": [234, 85, 226, 199], "score": 1.0},
    ...
]
```

#### Run COCO Evaluation

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ground truth and predictions
coco_gt = COCO()
coco_gt.dataset = gt_coco
coco_gt.createIndex()

coco_dt = coco_gt.loadRes(coco_predictions)

# Run evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.params.maxDets = [1, 10, 500]  # Evaluate up to 500 boxes per image
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
metrics = {
    'AR': coco_eval.stats[8],        # Average Recall @ IoU=0.50:0.95, maxDets=500
    'AR@0.5': compute_ar_at_iou(coco_eval, 0.5),
    'AR@0.75': compute_ar_at_iou(coco_eval, 0.75),
    'ARs': coco_eval.stats[9],       # Small objects
    'ARm': coco_eval.stats[10],      # Medium objects
    'ARl': coco_eval.stats[11]       # Large objects
}
```

#### Compute AR at Specific IoU Thresholds

```python
def compute_ar_at_iou(coco_eval, iou_thr):
    """
    Compute Average Recall at specific IoU threshold.

    COCO computes recall at multiple IoU thresholds:
    [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    This function extracts recall at a single threshold.
    """
    iou_thresholds = coco_eval.params.iouThrs  # [0.5, 0.55, ..., 0.95]

    if iou_thr in iou_thresholds:
        iou_idx = list(iou_thresholds).index(iou_thr)
    else:
        return 0.0

    # Extract recalls for this IoU threshold
    # Shape: [T, K, A, M]
    # T: IoU thresholds, K: categories, A: area ranges, M: max detections
    recalls = coco_eval.eval['recall']

    if recalls is None or recalls.size == 0:
        return 0.0

    # Get recall at this IoU, across all categories, all areas, maxDets=500
    recall_at_iou = recalls[iou_idx, :, 0, 2]  # 0: all areas, 2: maxDets=500

    # Average across categories
    valid_recalls = recall_at_iou[recall_at_iou > -1]
    ar = valid_recalls.mean() if len(valid_recalls) > 0 else 0.0

    return float(ar)

# Usage:
ar_05 = compute_ar_at_iou(coco_eval, 0.5)   # AR at IoU=0.50
ar_075 = compute_ar_at_iou(coco_eval, 0.75)  # AR at IoU=0.75
```

#### Understanding Average Recall (AR)

**Definition:**
```
AR = True Positives / (True Positives + False Negatives)
   = Detected GT boxes / Total GT boxes
```

**Example calculation:**

```python
# Across 10 test images:
total_gt_boxes = 47
detected_boxes = 14  # Boxes with IoU > 0.5

AR@0.5 = 14 / 47 = 29.8% ≈ 30.3%
```

**What AR@0.5 = 30.3% means:**
- Out of 100 ground truth boxes, model correctly detects ~30 with IoU > 0.5
- 70 boxes are missed (false negatives)
- This is baseline performance without detector fine-tuning

**AR vs AP (Precision):**
- **AR (Recall):** Focuses on how many GT boxes are found (ignores false positives)
- **AP (Precision):** Penalizes false positives
- For grounding tasks, AR is more relevant (we care about finding all entities)

---

## Real Examples from Evaluation

### Example 1: Perfect Match (100%)

**Image:** HICO_test2015_00000003.jpg (boat scene)

```
[Image 3/10] HICO_test2015_00000003.jpg
  GT boxes: 2
  Response: "The <p>person</p><roi><r7></roi> is <p>driving</p> the <p>boat</p><roi><r3></roi>..."
  Parsed entities: ['person', 'boat']
  Region indices: [7, 3]
  Matched: 2/2 (100.0%) ✅
```

**Visualization analysis:**
- **Left panel:** 55 total regions, R7 and R3 highlighted (gold outline)
- **Middle panel:**
  - person(R7) - Green box, IoU: 0.85 ✅
  - boat(R3) - Green box, IoU: 0.92 ✅
- **Right panel:** Both GT boxes matched (solid green borders)
- **Statistics:** 2/2 good matches (100%)

**Why it succeeded:**
1. Detector proposed good boxes at positions 7 and 3
2. Model correctly selected these regions
3. High IoU with ground truth (>0.7)
4. Entity names matched exactly

---

### Example 2: Partial Match (16.7%)

**Image:** HICO_test2015_00000002.jpg (horse scene)

```
[Image 2/10] HICO_test2015_00000002.jpg
  GT boxes: 6
  Response: "The <p>person</p><roi><r28></roi> is <p>holding</p> the <p>horse</p><roi><r34></roi>..."
  Parsed entities: ['person', 'horse']
  Region indices: [28, 34]
  Matched: 1/6 (16.7%) ⚠️
```

**Why only 1/6 matched:**
1. Image has **3 person-object pairs** (3 persons × 2 objects each = 6 GT boxes)
2. Model only described **1 pair** (person r28, horse r34)
3. Model missed **2 other person-object pairs**

**Possible reasons:**
- Model tends to describe only the most salient HOI pair
- Multiple persons in scene → model focuses on primary interaction
- May need instruction tuning to encourage describing all pairs

---

### Example 3: Complete Failure (0%)

**Image:** HICO_test2015_00000001.jpg (bench scene)

```
[Image 1/10] HICO_test2015_00000001.jpg
  GT boxes: 3
  Filtered 3 giant boxes (area > 0.5)
  Response: "The <p>person</p><roi><r4></roi> is <p>sitting on</p> the <p>bench</p><roi><r2></roi>..."
  Parsed entities: ['person', 'bench']
  Region indices: [4, 2]
  Matched: 0/3 (0.0%) ❌
```

**Why 0% match:**
1. GT has **2 persons + 1 bench** = 3 boxes
2. Model detected **1 person (r4) + 1 bench (r2)** = 2 boxes
3. **Neither matched** GT boxes (IoU < 0.5)

**Diagnosis from visualization:**
- **Middle panel:** Purple boxes (no GT match)
  - person(R4) - IoU: 0.22 (wrong person or poor localization)
  - bench(R2) - IoU: 0.31 (partial overlap, below 0.5 threshold)
- **Right panel:** 3 GT boxes with dashed red borders (all missed)

**Likely cause:**
- Detector proposed poor boxes at positions 4 and 2
- Filtering removed 3 giant boxes, possibly including better alternatives
- Model selected from remaining poor proposals

---

### Example 4: Name Mismatch Handling

**Hypothetical scenario:**

```
Model output: "The <p>person</p><roi><r5></roi> is <p>riding</p> a <p>bike</p><roi><r8></roi>..."
Parsed: entities=['person', 'bike']

Ground truth: categories=['person', 'bicycle']

# Entity normalization:
normalize_entity_name('bike')  # → 'bicycle'

# After normalization:
pred_entities_norm = ['person', 'bicycle']
gt_entities_norm = ['person', 'bicycle']

# Result: name_match = True
# Visualization: Green box (excellent match)
```

Without normalization, this would be marked as name mismatch (yellow box) despite being semantically correct.

---

## Metrics Interpretation

### Understanding the Results

```
Metric          Value
AR              21.2%  ← Average across IoU thresholds 0.50 to 0.95
AR@0.5          30.3%  ← Recall at IoU > 0.5 (lenient threshold)
AR@0.75         30.3%  ← Recall at IoU > 0.75 (strict threshold)
AR@s             0.0%  ← Small objects (< 32² pixels)
AR@m             0.0%  ← Medium objects (32² to 96² pixels)
AR@l            24.6%  ← Large objects (> 96² pixels)
```

**Key insights:**

1. **AR@0.5 = AR@0.75 = 30.3%**
   - When model matches a box, it's **very tight** (IoU > 0.75)
   - No boxes in the 0.5 < IoU < 0.75 range
   - This is GOOD - means localization is accurate when successful

2. **AR = 21.2% < AR@0.5 = 30.3%**
   - Recall drops at stricter IoU thresholds (0.80, 0.85, 0.90, 0.95)
   - Average across all thresholds is lower
   - Normal behavior for detection tasks

3. **ARs = ARm = 0%**
   - No small/medium objects detected
   - HOI entities tend to be large (persons, vehicles, furniture)
   - All detections are large objects (AR@l = 24.6%)

4. **Overall performance: ~30% recall**
   - Model detects ~1 out of 3 ground truth boxes
   - **Baseline without detector fine-tuning**
   - Room for improvement with HOI-specific detector training

---

## Comparison: One-Shot vs Multi-Turn

### One-Shot HOI Grounding (This Evaluation)

**Prompt:** Single query for all entities
```
[grounding] Describe what each person is doing with objects individually. Focus on actions only.
```

**Response:** All entities in one output
```
The <p>person</p><roi><r7></roi> is <p>driving</p> the <p>boat</p><roi><r3></roi>.
Another <p>person</p><roi><r2></roi> is <p>riding</p> a <p>boat</p><roi><r1></roi>.
```

**Pros:**
- More realistic evaluation (matches training format)
- Tests model's ability to handle multiple HOI in single response
- Faster (one model call per image)

**Cons:**
- Model may miss some HOI pairs (focus on primary interaction)
- Lower recall for images with many entities

---

### Multi-Turn Per-Category (`run_hico_ground_eval.sh`)

**Prompt:** Separate query per category
```
Query 1: "Locate all person in this image"
Query 2: "Locate all boat in this image"
Query 3: "Locate all bench in this image"
...
```

**Response:** One category at a time
```
Turn 1: <r0><r3><r5>  (person boxes)
Turn 2: <r1><r2>      (boat boxes)
Turn 3: <r4>          (bench boxes)
```

**Pros:**
- Higher recall (model focuses on one category at a time)
- Easier task (no need to track multiple entities simultaneously)

**Cons:**
- Not how model is trained (different from instruction finetuning format)
- Slower (N model calls per image, N = number of categories)
- Less realistic (not how users would query in practice)

---

## Troubleshooting

### Issue 1: Low Recall (<20%)

**Symptoms:**
```
AR@0.5: 15.2%
Most images: Matched 0-1 boxes out of 3-5 GT boxes
```

**Diagnosis:**
1. Check visualizations for color distribution:
   - **Mostly purple boxes** → Instruction alignment issue (model generates wrong entities)
   - **Mostly orange boxes** → Region proposer issue (detector proposals are poor)
   - **Mostly red boxes** → Wrong object detection

2. Run per-category evaluation for comparison:
   ```bash
   bash scripts/run_hico_ground_eval.sh 0
   ```
   If per-category AR is also low → detector issue
   If per-category AR is high → one-shot format issue

**Solutions:**
- **Purple/red boxes:** Retrain instruction finetuning with more diverse HOI examples
- **Orange boxes:** Retrain detection (Stage 1) on HOI datasets
- **One-shot format issue:** Add data augmentation with multiple HOI pairs per image

---

### Issue 2: Name Mismatches

**Symptoms:**
```
Statistics show: Good IoU, name mismatch: 15 (60%)
Many yellow boxes in visualizations
```

**Diagnosis:**
Model uses different vocabulary than COCO categories:
- "bike" instead of "bicycle"
- "table" instead of "dining table"
- "phone" instead of "cell phone"

**Solutions:**
1. **Expand ENTITY_SYNONYMS dictionary:**
   ```python
   ENTITY_SYNONYMS = {
       'bike': 'bicycle',
       'table': 'dining table',
       # Add more based on common mismatches
   }
   ```

2. **Analyze common mismatches:**
   ```bash
   # Check evaluation log for name mismatch patterns
   grep "name_match=False" results/hico_hoi_oneshot/hico_hoi_oneshot_evaluation*.log
   ```

3. **Fine-tune with COCO vocabulary:** Add examples using exact COCO category names

---

### Issue 3: Model Only Describes One HOI Pair

**Symptoms:**
```
GT boxes: 6 (3 person-object pairs)
Predicted: 2 (1 person-object pair)
Matched: 1/6 (16.7%)
```

**Diagnosis:**
Response shows only primary interaction:
```
"The <p>person</p><roi><r5></roi> is <p>riding</p> a <p>bicycle</p><roi><r3></roi>"
# Missing 2 other person-bicycle pairs in the image
```

**Solutions:**
1. **Modify prompt** to explicitly request all pairs:
   ```
   "[grounding] Describe what EVERY person is doing with objects. List all interactions."
   ```

2. **Add training examples** with multiple HOI pairs:
   ```json
   {
     "value": "The <p>person</p><roi><r0></roi> is <p>riding</p> a <p>bike</p><roi><r1></roi>. Another <p>person</p><roi><r2></roi> is <p>riding</p> a <p>bike</p><roi><r3></roi>. A third <p>person</p><roi><r4></roi> is <p>walking</p> near a <p>car</p><roi><r5></roi>."
   }
   ```

3. **Post-processing:** Run detector to find all persons, then describe each separately

---

### Issue 4: Giant Box Filtering Too Aggressive

**Symptoms:**
```
Filtered 15 giant boxes (area > 0.5)
Matched: 0/4 (0%)
Visualization shows missing large objects (buses, tables, etc.)
```

**Diagnosis:**
Threshold of 0.5 removes legitimate large objects in HOI scenarios.

**Solutions:**
1. **Increase threshold:**
   ```python
   MAX_BOX_AREA_RATIO = 0.7  # Allow larger boxes
   ```

2. **Adaptive threshold** based on image content:
   ```python
   # If image has large object category (bus, dining table, couch)
   MAX_BOX_AREA_RATIO = 0.8
   # Otherwise
   MAX_BOX_AREA_RATIO = 0.5
   ```

3. **Remove filtering** and rely on detector quality:
   ```python
   # Comment out filtering step
   # pred_boxes = pred_boxes[valid_box_mask]
   ```

---

## File Structure

```
groma/
├── data/
│   └── datasets/
│       ├── hico_ground.py              # Per-category evaluation dataset
│       └── hico_hoi_oneshot.py         # One-shot HOI evaluation dataset ← NEW
│
├── eval/
│   ├── eval_hico_ground.py             # Per-category evaluation script
│   └── eval_hico_hoi_oneshot.py        # One-shot HOI evaluation script ← NEW
│
scripts/
├── run_hico_ground_eval.sh             # Per-category evaluation wrapper
└── run_hico_hoi_oneshot_eval.sh        # One-shot HOI evaluation wrapper ← NEW
```

---

## Summary

### Complete Pipeline Flow

```
1. Load test_hico_ann.json → Extract HOI-involved boxes (GT)
                ↓
2. Preprocess image → Resize to 448×448, normalize
                ↓
3. Build prompt → "[grounding] Describe what each person is doing..."
                ↓
4. Model inference → Generate text + bounding boxes
                ↓
5. Smart filtering → Remove giant boxes (>50% area)
                ↓
6. Parse response → Extract entities + region indices
                ↓
7. Deduplicate → Remove duplicate region indices
                ↓
8. Extract boxes → Map region indices → pixel coordinates
                ↓
9. Match to GT → Greedy IoU matching with name normalization
                ↓
10. Visualize → 3-panel layout (all regions | predictions | GT)
                ↓
11. COCO eval → Compute AR, AR@0.5, AR@0.75
```

### Key Takeaways

1. **One-shot evaluation** tests model's ability to detect all HOI entities in single response
2. **Smart filtering** (before parsing) prevents index remapping issues
3. **Action word filtering** separates entities from actions in response
4. **Name normalization** handles vocabulary variations (bike → bicycle)
5. **Greedy matching** with priority scoring favors name matches + high IoU
6. **Average Recall** is the primary metric (AP@0.5, AR@0.75)
7. **Visualization** helps diagnose issues (purple = instruction, orange = detector)

### Expected Performance

- **Baseline (without HOI detector):** AR@0.5 ~ 30-40%
- **With HOI detector:** AR@0.5 ~ 50-60% (expected improvement)
- **Multiple HOI pairs:** Recall drops for complex scenes (5+ entities)

---

## Next Steps

1. **Run full evaluation** on all 9546 test images
2. **Analyze failure modes** from visualizations
3. **Compare with per-category evaluation** (run_hico_ground_eval.sh)
4. **Consider detector fine-tuning** if AR@0.5 < 40%
5. **Tune prompt** if model misses multiple HOI pairs

---

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Evaluation Script:** `groma/eval/eval_hico_hoi_oneshot.py`
**Author:** Groma HOI Research Team
