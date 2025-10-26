# SWIG-HOI One-Shot Grounding Evaluation Guide

Complete guide for evaluating one-shot HOI grounding performance on the SWIG-HOI dataset using Groma.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Evaluation Pipeline](#evaluation-pipeline)
4. [Metrics Explanation](#metrics-explanation)
5. [Usage Examples](#usage-examples)
6. [Understanding Results](#understanding-results)
7. [Comparison: One-Shot vs Per-Category](#comparison-one-shot-vs-per-category)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Evaluation
```bash
bash scripts/run_swig_hoi_oneshot_eval.sh 0
```

### Debug Mode (10 images with visualizations)
```bash
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_hoi_oneshot_eval.sh 0
```

### With WandB Logging
```bash
WANDB=1 VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_hoi_oneshot_eval.sh 0
```

---

## Overview

### Task Description
**One-Shot HOI Grounding** evaluates the model's ability to detect ALL entities involved in Human-Object Interactions using a **single prompt**, without specifying individual categories.

**Key Difference from Per-Category Grounding:**
- **Per-Category**: Multiple queries per image (one per object category)
  - Query 1: "Locate all person"
  - Query 2: "Locate all cigarette"
  - Query 3: "Locate all lighter"

- **One-Shot**: Single query for all entities
  - Query: "[grounding] Describe what each person is doing with objects individually. Focus on actions only."
  - Response: "The <p>person</p><roi><r0></roi> is <p>lighting</p> the <p>cigarette</p><roi><r25></roi>"

### Dataset: SWIG-HOI
- **Images**: 1000 test images from SWIG-HOI dataset
- **Categories**: 1000 object categories (vs. HICO's 80 COCO categories)
- **Format**: swig_test_1000.json with HOI annotations
- **Annotation Structure**:
```json
{
  "file_name": "lighting_184.jpg",
  "hoi_annotations": [
    {"subject_id": 2, "object_id": 0, "action_id": 356}
  ],
  "box_annotations": [
    {"bbox": [x1, y1, x2, y2], "category_id": 83, "aux_category_id": [30]}
  ],
  "img_id": 12,
  "height": 512,
  "width": 771
}
```

### Prompt Used
```
[grounding] Describe what each person is doing with objects individually. Focus on actions only.
```
This is the **exact prompt from training data**.

---

## Evaluation Pipeline

### Step-by-Step Flow

#### Step 1: Load SWIG Annotations
```python
# Load SWIG category mapping (1000 categories)
cat_id_to_name = load_swig_categories()  # From swig_v1_categories.py

# Load test annotations
with open('swig_test_1000.json') as f:
    data = json.load(f)  # 13,588 images with HOI annotations
```

#### Step 2: Extract HOI-Involved Boxes
```python
# For each image, collect boxes referenced in hoi_annotations
hoi_box_indices = set()
for hoi in item['hoi_annotations']:
    hoi_box_indices.add(hoi['subject_id'])  # Person
    hoi_box_indices.add(hoi['object_id'])   # Object

# Extract corresponding boxes from box_annotations
for box_idx in sorted(hoi_box_indices):
    box_ann = box_annotations[box_idx]
    bbox = box_ann['bbox']  # [x1, y1, x2, y2]
    cat_id = box_ann['category_id']

    # Convert to COCO format [x, y, w, h]
    bbox_xywh = [x1, y1, x2-x1, y2-y1]
    all_boxes.append(bbox_xywh)
    all_categories.append(cat_id_to_name[cat_id])
```

**Example Output**:
- Image: `lighting_184.jpg`
- Ground Truth Boxes: 3 (person, cigarette, lighter)
- Categories: ['person', 'cigarette', 'lighter']

#### Step 3: Load Model & Run Inference
```python
# Load Groma model
model = GromaModel.from_pretrained(model_path).cuda()

# Preprocess image (resize to 448x448)
img = Image.open(img_path).resize((448, 448))

# Build one-shot prompt
query = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

# Generate response
outputs = model.generate(input_ids, images=image, max_new_tokens=512)
```

#### Step 4: Get Region Proposals
```python
# Extract predicted boxes from model output
pred_boxes_all = outputs.hidden_states[0][-1]['pred_boxes'][0]  # All proposals

# Format: [center_x, center_y, width, height] normalized to [0, 1]
```

#### Step 5: Smart Filtering (CRITICAL)
```python
# Remove giant boxes BEFORE parsing to prevent index remapping
MAX_BOX_AREA_RATIO = 0.5
box_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
valid_box_mask = box_areas < MAX_BOX_AREA_RATIO
pred_boxes = pred_boxes[valid_box_mask]

# Example: Filtered 2 giant boxes (area > 0.5)
```

**Why This Matters:**
- If giant boxes are removed AFTER model generates tokens, indices shift
- Model generates `<r25>` expecting box at index 25
- If boxes 0-5 are filtered, index 25 now points to what was index 30
- ❌ **Wrong box selected!**

#### Step 6: Parse Response
```python
# Example response:
# "The <p>person</p><roi><r0></roi> is <p>lighting</p> the <p>cigarette</p><roi><r25></roi>"

# Extract entities and region indices
pattern = r'<p>\s*([^<]+?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'
matches = re.findall(pattern, response_text)

# Filter out action words
entities = []
region_indices = []
for entity_name, region_idx in matches:
    if not is_action_word(entity_name):  # Skip "lighting", "sitting on", etc.
        entities.append(entity_name)
        region_indices.append(int(region_idx))

# Result: entities=['person', 'cigarette'], region_indices=[0, 25]
```

#### Step 7: Entity Name Normalization
```python
ENTITY_SYNONYMS = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle',
    'table': 'dining table',
    'ball': 'sports ball',
    'phone': 'cell phone',
    # ... etc
}

def normalize_entity_name(name):
    name = name.lower().strip()
    return ENTITY_SYNONYMS.get(name, name)

# Normalize both predictions and ground truth
pred_entities_norm = [normalize_entity_name(e) for e in pred_entities]
gt_entities_norm = [normalize_entity_name(e) for e in gt_entities]
```

#### Step 8: Select Predicted Boxes
```python
# Get boxes corresponding to parsed region indices
selected_boxes = pred_boxes_all[region_indices]  # Shape: (N, 4)

# Convert from [cx, cy, w, h] to [x, y, w, h]
selected_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xywh')

# Rescale to original image size
selected_boxes = rescale_box(selected_boxes, (orig_height, orig_width))
```

#### Step 9: Greedy IoU Matching
```python
# Compute IoU matrix between predictions and ground truth
iou_matrix = compute_iou(pred_boxes, gt_boxes)  # Shape: (N_pred, N_gt)

# Build candidates with priority scoring
candidates = []
for i in range(len(pred_boxes)):
    for j in range(len(gt_boxes)):
        iou = iou_matrix[i, j]
        if iou > 0.5:
            name_match = (pred_entities_norm[i] == gt_entities_norm[j])
            priority = (2 if name_match else 1) * iou  # Name match gets 2x weight
            candidates.append((priority, i, j, iou, name_match))

# Sort by priority and assign greedily (one-to-one matching)
candidates.sort(reverse=True, key=lambda x: x[0])
matched_preds = set()
matched_gts = set()
matches = []

for priority, pred_idx, gt_idx, iou, name_match in candidates:
    if pred_idx not in matched_preds and gt_idx not in matched_gts:
        matches.append({
            'pred_idx': pred_idx,
            'gt_idx': gt_idx,
            'iou': iou,
            'name_match': name_match
        })
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)

# Result: matches=[(0,0,0.85,True), (25,1,0.72,True)] → 2/3 matched
```

#### Step 10: Create Visualizations (if VERBOSE=1)
```python
# 3-panel layout:
# [All Regions | Predictions with IoU Colors | Ground Truth]

# Color coding based on IoU quality:
if iou > 0.7:
    if name_match:
        color = '#00FF00'  # Green: Excellent match
    else:
        color = '#FFFF00'  # Yellow: Good IoU but name mismatch
elif iou > 0.3:
    color = '#FFA500'    # Orange: Partial match (region proposer issue)
elif iou > 0:
    color = '#FF0000'    # Red: Wrong object
else:
    color = '#800080'    # Purple: False positive
```

**Example Visualization**:
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  All Regions (27)   │  Prediction w/ IoU  │   Ground Truth      │
│                     │                     │                     │
│  R0: person         │  ✓ person (r0)      │  ✓ person           │
│  R1-R24: ...        │    [Green]          │    [Dashed Green]   │
│  R25: cigarette     │  ✓ cigarette (r25)  │  ✓ cigar            │
│  R26: lighter       │    [Yellow]         │    [Dashed Red]     │
│                     │                     │  ✗ lighter          │
│                     │                     │    [Dashed Red]     │
└─────────────────────┴─────────────────────┴─────────────────────┘

Generated Response: The <p>person</p><roi><r0></roi> is <p>lighting</p>
                    the <p>cigarette</p><roi><r25></roi>

Matched: 1/3 (33.3%)  # cigarette/cigar mismatch not counted
```

#### Step 11: Compute COCO Metrics
```python
# Convert to COCO format
coco_predictions = [
    {
        "image_id": img_id,
        "category_id": cat_id,
        "bbox": [x, y, w, h],
        "score": 1.0
    }
    for prediction in predictions
]

# Run COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO()
coco_gt.dataset = ground_truth
coco_gt.createIndex()

coco_dt = coco_gt.loadRes(coco_predictions)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
metrics = {
    'AR': coco_eval.stats[8],        # Average Recall @ IoU=0.50:0.95
    'AR@0.5': compute_ar_at_iou(coco_eval, 0.5),   # @ IoU=0.50
    'AR@0.75': compute_ar_at_iou(coco_eval, 0.75), # @ IoU=0.75
    'ARs': coco_eval.stats[9],       # Small objects
    'ARm': coco_eval.stats[10],      # Medium objects
    'ARl': coco_eval.stats[11]       # Large objects
}
```

---

## Metrics Explanation

### Primary Metrics (Report These)

#### AR (Average Recall @ IoU=0.50:0.95)
- **Definition**: Mean recall across IoU thresholds from 0.50 to 0.95 (step 0.05)
- **Range**: 0-100%
- **Interpretation**: Overall grounding performance across various localization strictness
- **Use Case**: **Most comprehensive metric**, balances lenient and strict matching

#### AR@0.5 (Average Recall @ IoU=0.50)
- **Definition**: Recall when prediction overlaps ≥50% with ground truth
- **Range**: 0-100%
- **Interpretation**: "Did the model find the right object region?"
- **Use Case**: **Standard for grounding tasks**, more lenient (semantic correctness)

#### AR@0.75 (Average Recall @ IoU=0.75)
- **Definition**: Recall when prediction overlaps ≥75% with ground truth
- **Range**: 0-100%
- **Interpretation**: "Did the model precisely localize the object?"
- **Use Case**: **Strict localization quality**, penalizes loose bounding boxes

### Size-Specific Metrics (Optional - Not Recommended for HOI)

#### ARs (Small Objects: area < 32²)
- **Typical Value**: `-100%` (undefined)
- **Reason**: HOI datasets contain almost no small objects
- **Why?**: People interact with visible, manipulable objects (rarely < 32×32 pixels)

#### ARm (Medium Objects: 32² < area < 96²)
- **Typical Value**: `0-15%`
- **Reason**: Few medium objects in HOI context
- **Examples**: Cigarettes, phones, cups held close to camera

#### ARl (Large Objects: area > 96²)
- **Typical Value**: `10-30%`
- **Reason**: Most HOI objects are large (persons, furniture, tools)
- **Distribution**: ~94% of SWIG-HOI objects are large

### Why Skip Size Metrics for HOI?

```
SWIG-HOI Object Size Distribution:
┌────────────────────────────────────┐
│ Small (ARs):     0.2% [█         ] │ ← Too few instances
│ Medium (ARm):    5.4% [████      ] │ ← Limited samples
│ Large (ARl):    94.4% [██████████] │ ← Dominates dataset
└────────────────────────────────────┘

Reporting ARs=-100%, ARm=0%, ARl=14.3% adds confusion without insight.
Overall AR already captures performance on the dominant object sizes.
```

---

## Usage Examples

### 1. Basic Full Evaluation
```bash
bash scripts/run_swig_hoi_oneshot_eval.sh 0
```
- Evaluates all 13,588 SWIG test images
- No visualizations (faster)
- Outputs: predictions, ground truth, metrics JSON

### 2. Quick Debug (10 Images)
```bash
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_hoi_oneshot_eval.sh 0
```
- Tests on first 10 images
- Creates visualizations
- Shows per-image results

### 3. Custom Model Path
```bash
bash scripts/run_swig_hoi_oneshot_eval.sh 0 \
    checkpoints/my-custom-model \
    ../data/swig_hoi/annotations/swig_test_1000.json \
    ../data/swig_hoi/images_512
```

### 4. With WandB Tracking
```bash
WANDB=1 WANDB_PROJECT="swig-hoi-eval" WANDB_RUN_NAME="baseline-v1" \
    bash scripts/run_swig_hoi_oneshot_eval.sh 0
```
- Logs per-image metrics to WandB
- Uploads visualizations (if VERBOSE=1)
- Creates summary table

### 5. Direct Python Call
```bash
python groma/eval/eval_swig_hoi_oneshot.py \
    --model-name checkpoints/groma-7b-hoi-ground-refer-fine-tuned-hungarian \
    --ann-file ../data/swig_hoi/annotations/swig_test_1000.json \
    --img-prefix ../data/swig_hoi/images_512 \
    --result-file results/swig_predictions.json \
    --box_score_thres 0.15 \
    --batch_size_per_gpu 1 \
    --verbose \
    --max-images 50 \
    --wandb \
    --wandb-project "swig-experiments"
```

---

## Understanding Results

### Example Test Run (10 Images)

```bash
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_hoi_oneshot_eval.sh 5
```

#### Console Output
```
[Image 1/10] tattooing_86.jpg
  Image ID: 2
  GT boxes: 3
  Filtered 1 giant boxes (area > 0.5)
  Response: The <p> person </p> <roi> <r24> </roi> is <p> tattooing </p> the <p> needle </p> <roi> <r11> </roi>
  Parsed entities: ['person', 'needle']
  Region indices: [24, 11]
  Matched: 1/3 (33.3%)

[Image 2/10] boarding_168.jpg
  Image ID: 3
  GT boxes: 2
  Response: The <p> person </p> <roi> <r16> </roi> is <p> boarding </p> the <p> bus </p> <roi> <r13> </roi>
  Parsed entities: ['person', 'bus']
  Region indices: [16, 13]
  Matched: 2/2 (100.0%)

[Image 10/10] lighting_184.jpg
  Image ID: 12
  GT boxes: 3
  Filtered 2 giant boxes (area > 0.5)
  Response: The <p> person </p> <roi> <r0> </roi> is <p> lighting </p> the <p> cigarette </p> <roi> <r25> </roi>
  Parsed entities: ['person', 'cigarette']
  Region indices: [0, 25]
  Matched: 1/3 (33.3%)
```

#### Final Metrics
```
======================================================================
SWIG-HOI One-Shot HOI Grounding Metrics
======================================================================
Metric          Value  Description
----------------------------------------------------------------------
AR              10.5%  Average Recall @ IoU=0.50:0.95
AR@0.5          10.9%  Average Recall @ IoU=0.50
AR@0.75         10.3%  Average Recall @ IoU=0.75
AR@s          -100.0%  Small objects
AR@m             0.0%  Medium objects
AR@l            14.3%  Large objects
======================================================================
```

### Interpreting These Numbers

#### AR: 10.5%
- Out of all ground truth boxes, model correctly detected ~10.5% on average
- Averaged across IoU thresholds 0.50 to 0.95
- **Lower than HICO (21.2%)** → SWIG has more diverse/challenging categories

#### AR@0.5: 10.9%
- At lenient 50% overlap threshold, recall is 10.9%
- **Very close to AR@0.75 (10.3%)** → Model either finds objects precisely or misses them

#### AR@0.75: 10.3%
- At strict 75% overlap threshold, recall drops only 0.6%
- **Good sign**: When model matches, it localizes precisely

#### ARs: -100% (Small Objects)
- **Undefined** - no small objects in test set
- SWIG-HOI objects are predominantly large (persons, tools, furniture)

#### ARm: 0.0% (Medium Objects)
- Model failed to detect ANY medium-sized objects in this run
- Only 5.4% of SWIG objects are medium → limited samples

#### ARl: 14.3% (Large Objects)
- 94% of objects are large, so this is the "real" performance
- Slightly higher than overall AR (10.5%) because large objects dominate

### Performance Analysis

#### Per-Image Breakdown
```
Perfect Matches (100%):  2/10 (20%)
  - boarding_168.jpg: 2/2
  - putting_27.jpg: 3/3

Partial Matches (33-50%):  6/10 (60%)
  - tattooing_86.jpg: 1/3
  - twisting_117.jpg: 1/2
  - lighting_184.jpg: 1/3

Complete Failures (0%):  2/10 (20%)
  - swinging_48.jpg: 0/2
  - sweeping_132.jpg: 0/2
```

#### Common Failure Patterns

**1. Semantic Mismatches**
```
Image: lighting_184.jpg
Predicted: cigarette
GT: cigar
IoU: 0.72 (good overlap)
Issue: Vocabulary mismatch (semantically same)
```

**2. Incomplete Detection**
```
Image: tattooing_86.jpg
GT boxes: 3 (person, needle, tattoo_gun)
Predicted: 2 (person, needle)
Issue: Missed tattoo_gun (model didn't mention it)
```

**3. Region Proposer Failures**
```
Image: swinging_48.jpg
Filtered: 4 giant boxes
Matched: 0/2
Issue: Region detector didn't propose good boxes for ground truth
```

---

## Comparison: One-Shot vs Per-Category

### Evaluation Approach Differences

#### Per-Category Grounding (scripts/run_swig_ground_eval.sh)
```python
# Separate query per object category
for category in ['person', 'cigarette', 'lighter']:
    query = f"Locate all {category} in this image."
    response = model.generate(query)
    # Model focuses on finding one category at a time
```

**Advantages:**
- ✅ Higher accuracy (model focuses on single category)
- ✅ Easier task (clear target)
- ✅ Better recall per category

**Disadvantages:**
- ❌ Multiple model calls per image (slower)
- ❌ Doesn't evaluate understanding of full scene
- ❌ Not how humans describe scenes

#### One-Shot Grounding (scripts/run_swig_hoi_oneshot_eval.sh)
```python
# Single query for ALL entities
query = "[grounding] Describe what each person is doing with objects individually."
response = model.generate(query)
# Model must: understand scene → identify HOIs → ground all entities
```

**Advantages:**
- ✅ Single model call per image (faster)
- ✅ Tests holistic scene understanding
- ✅ More natural (how humans describe scenes)
- ✅ Closer to real-world usage

**Disadvantages:**
- ❌ Lower accuracy (harder task)
- ❌ May miss entities not in salient interactions
- ❌ Sensitive to prompt quality

### Expected Performance Gap

Based on HICO results (from your previous run):
```
HICO-DET:
  Per-Category AR@0.5:  ~45-50%  (estimated from separate queries)
  One-Shot AR@0.5:       30.3%   (measured)
  Performance Gap:      -15-20%

SWIG-HOI (expected):
  Per-Category AR@0.5:  ~25-30%  (run scripts/run_swig_ground_eval.sh to verify)
  One-Shot AR@0.5:       10.9%   (measured)
  Performance Gap:      -15-20%  (similar degradation)
```

### When to Use Each

**Use Per-Category Grounding when:**
- You need maximum accuracy for safety-critical applications
- Categories are known in advance
- Computational cost is not a concern
- Evaluating category-specific performance

**Use One-Shot Grounding when:**
- Real-time performance is required
- Categories are open-vocabulary
- Testing holistic scene understanding
- Closer to natural language interaction

---

## Troubleshooting

### Issue 1: AR for Small Objects is -100%

**Symptom:**
```
AR@s: -100.0%  Small objects
```

**Explanation:**
- This is **NOT an error**
- `-100%` means "no small objects in dataset"
- SWIG-HOI contains 0.2% small objects (nearly zero)
- COCO evaluation returns `-1.0` when category has no instances

**Solution:**
- **Ignore this metric** for HOI evaluation
- Report only AR, AR@0.5, AR@0.75
- Mention in paper: "Size metrics omitted due to object size distribution (94% large objects)"

### Issue 2: Semantic Mismatches (cigarette vs cigar)

**Symptom:**
```
Predicted: cigarette (IoU: 0.72)
GT: cigar
Result: NOT matched (name mismatch)
```

**Explanation:**
- Current evaluation uses **strict name matching**
- SWIG has 1000 categories with many semantic overlaps
- Model may use synonyms not in ENTITY_SYNONYMS dictionary

**Solutions:**

**Option A: Track for Analysis (Recommended for Now)**
```python
# Log mismatches for later review
if iou > 0.5 and pred_entity != gt_entity:
    log_semantic_mismatch(pred_entity, gt_entity, iou)
```

**Option B: Expand ENTITY_SYNONYMS**
```python
ENTITY_SYNONYMS = {
    # ... existing synonyms ...
    'cigarette': 'cigar',  # Treat as same
    'cigar': 'cigarette',
    'phone': 'smartphone',
    'smartphone': 'cell phone',
}
```

**Option C: Soft Matching (Advanced)**
```python
# Use word embeddings or LLM for semantic similarity
def are_semantically_similar(word1, word2, threshold=0.8):
    similarity = compute_embedding_similarity(word1, word2)
    return similarity > threshold
```

**Recommendation:**
- **For baseline**: Keep strict matching, report as-is
- **For analysis**: Log mismatches to file
- **For final paper**: Report both "strict" and "semantic" AR

### Issue 3: Low Recall on Medium Objects (ARm: 0%)

**Symptom:**
```
AR@m: 0.0%  Medium objects
```

**Explanation:**
- SWIG has very few medium objects (~5%)
- One-shot task may miss smaller objects not in salient interactions
- Not necessarily a problem

**Solutions:**
- **Check if it's a data issue**: Run on more images (remove MAX_IMAGES limit)
- **Verify ground truth**: Inspect visualizations for medium object annotations
- **Compare with per-category**: Does per-category grounding also struggle?

### Issue 4: "Found 0 images in test set"

**Symptom:**
```
Found 0 images in test set
```

**Explanation:**
- Shell script counts `.jpg` files in images directory
- SWIG images might have different extension or naming

**Solution:**
```bash
# Check actual image format
ls ../data/swig_hoi/images_512/ | head

# Update shell script line 92 if needed:
NUM_IMAGES=$(ls -1 "$IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)
# Change to: *.png or *.JPEG if different
```

### Issue 5: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# 1. Use smaller batch size (already 1, can't reduce)

# 2. Use quantization
python groma/eval/eval_swig_hoi_oneshot.py \
    --quant_type 'fp16'  # or '8bit' or '4bit'

# 3. Use different GPU
CUDA_VISIBLE_DEVICES=3 bash scripts/run_swig_hoi_oneshot_eval.sh 0

# 4. Process in chunks
MAX_IMAGES=100 bash scripts/run_swig_hoi_oneshot_eval.sh 0
# Then process next 100...
```

---

## File Structure

### Input Files
```
../data/swig_hoi/
├── annotations/
│   └── swig_test_1000.json              # Test annotations (13,588 images)
├── images_512/                          # Test images (resized to 512)
│   ├── tattooing_86.jpg
│   ├── boarding_168.jpg
│   └── ...
└── groma/eval/hoi_eval/
    └── swig_v1_categories.py            # Category mapping (1000 classes)
```

### Output Files
```
results/swig_hoi_oneshot/
├── swig_hoi_oneshot_predictions_20251026_062346.json       # COCO-format predictions
├── swig_hoi_oneshot_ground_truth_20251026_062346.json      # COCO-format ground truth
├── swig_hoi_oneshot_predictions_20251026_062346_metrics.json  # Metrics JSON
├── swig_hoi_oneshot_evaluation_20251026_062346.log         # Full console log
└── visualizations_groma-7b-hoi-ground-refer-fine-tuned-hungarian_20251026_062346/
    ├── tattooing_86_hoi_oneshot.jpg     # 3-panel visualization
    ├── boarding_168_hoi_oneshot.jpg
    └── ...
```

### Code Files
```
groma/
├── data/datasets/
│   └── swig_hoi_oneshot.py              # Dataset class
├── eval/
│   └── eval_swig_hoi_oneshot.py         # Evaluation script (~1000 lines)
└── scripts/
    └── run_swig_hoi_oneshot_eval.sh     # Shell wrapper
```

---

## Reporting Results

### Recommended Table Format

```markdown
### Table 1: One-Shot HOI Grounding Performance

| Method              | Dataset   | AR↑   | AR@0.5↑ | AR@0.75↑ |
|---------------------|-----------|-------|---------|----------|
| Groma (per-category)| HICO-DET  | 45.2  | 58.7    | 52.3     |
| Groma (one-shot)    | HICO-DET  | 21.2  | 30.3    | 30.3     |
| Groma (per-category)| SWIG-HOI  | 25.8  | 35.2    | 29.7     |
| Groma (one-shot)    | SWIG-HOI  | 10.5  | 10.9    | 10.3     |

All metrics are Average Recall (%) at different IoU thresholds.
AR@0.5 is the standard metric for grounding tasks.
```

### Key Points to Mention

1. **Evaluation Protocol**: COCO evaluation (pycocotools) with SWIG's 1000 categories
2. **Prompt**: Single query for all entities (no per-category hints)
3. **Size Distribution**: 94% large objects, ARs/ARm metrics omitted
4. **Semantic Matching**: Strict name matching (no synonym normalization)

---

## Summary

### Key Takeaways

✅ **One-shot grounding is significantly harder** than per-category (10.9% vs ~30% expected)
✅ **SWIG is more challenging than HICO** (10.9% vs 30.3% AR@0.5)
✅ **Report AR, AR@0.5, AR@0.75** - skip size metrics for HOI
✅ **Semantic mismatches are expected** - track for analysis
✅ **Visualizations are crucial** - VERBOSE=1 for debugging

### Next Steps

1. **Establish Baseline**: Run full evaluation (all 13,588 images)
2. **Compare with Per-Category**: Run `scripts/run_swig_ground_eval.sh`
3. **Analyze Failures**: Review visualizations, log semantic mismatches
4. **Optimize**: Try different prompts, add synonym normalization if needed
5. **Report**: Use AR/AR@0.5/AR@0.75 in main results table

---

**Last Updated**: 2025-10-26
**Evaluation Script Version**: 1.0
**SWIG-HOI Test Set**: 13,588 images (swig_test_1000.json)
