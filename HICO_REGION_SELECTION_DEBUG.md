# HICO Region Selection Debug Guide

## Problem Statement

The HICO grounding evaluation shows that the model is able to generate correct regions (10 regions detected), but the IoU selection process is choosing the **same region multiple times** instead of selecting different regions.

Example from your output:
```
[Image 3/10] HICO_test2015_00000003.jpg
  Categories in image: ['person', 'boat']

  [person] ✓ SUCCESS
    Predicted: 2 boxes
    Selected regions: R9, R9  ← DUPLICATE!
      Box 1: [230.3, 149.1, 209.5, 141.1]
      Box 2: [230.3, 149.1, 209.5, 141.1]  ← SAME BOX!

  [boat] ✓ SUCCESS
    Predicted: 2 boxes
    Selected regions: R2, R2  ← DUPLICATE!
      Box 1: [230.3, 149.1, 209.5, 141.1]
      Box 2: [230.3, 149.1, 209.5, 141.1]  ← SAME BOX!
```

## Root Cause Analysis

The issue is **NOT with the IoU calculation** - it's with the **model's token generation**. The model is generating duplicate box tokens like `<r#9><r#9>` instead of `<r#9><r#8>`.

### Code Flow

1. **Region Proposal** (lines 665-666 in [eval_hico_ground.py](groma/eval/eval_hico_ground.py:665-666)):
   ```python
   pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
   # This gives you 10 regions: R0, R1, R2, ..., R9
   ```

2. **Token Generation** (lines 664, 673-682):
   ```python
   output_ids = outputs.sequences
   # Model generates text with box tokens like: <r#9><r#9>

   predicted_box_tokens = [
       id for id in output_ids[0, input_token_len:]
       if id in model.box_idx_token_ids
   ]
   # This extracts the token IDs: [token_for_9, token_for_9]

   selected_box_inds = [
       model.box_idx_token_ids.index(id)
       for id in predicted_box_tokens
   ]
   # This maps to region indices: [9, 9] ← DUPLICATES!
   ```

3. **Box Selection** (lines 707-711):
   ```python
   selected_boxes = pred_boxes[selected_box_inds]
   # This selects: [pred_boxes[9], pred_boxes[9]]
   # Result: Same box twice!
   ```

## Debug Enhancements Added

### 1. Token Analysis Debug Output

Added detailed token debugging (lines 694-710):
```python
if show_verbose:
    print(f"\n    🔍 DEBUG - Token Analysis for [{category}]:")
    print(f"       Generated text: {generated_text}")
    print(f"       Box tokens found: {predicted_box_tokens}")
    print(f"       Mapped to region indices: {selected_box_inds}")
    print(f"       Total available regions: {len(pred_boxes)}")

    # Check for duplicates
    duplicates = Counter(selected_box_inds)
    duplicate_regions = [idx for idx, count in duplicates.items() if count > 1]
    if duplicate_regions:
        print(f"       ⚠️  DUPLICATE REGIONS DETECTED: {duplicate_regions}")
        for dup_idx in duplicate_regions:
            dup_count = duplicates[dup_idx]
            print(f"          Region R{dup_idx} selected {dup_count} times")
```

### 2. Enhanced Box Output

Updated to show region indices with each box (lines 756-762):
```python
for i, (box, reg_idx) in enumerate(zip(selected_boxes, selected_box_inds)):
    print(f"      Box {i+1} (R{reg_idx}): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
```

### 3. Visualization Enhancements

Added token debug info to visualizations (lines 347-370):
- Shows generated tokens for each category
- Highlights duplicates in RED with format: `[DUPLICATES: R9×2]`
- Displays in the legend area of the visualization image

### 4. Debug Script

Created [scripts/debug_hico_regions.sh](scripts/debug_hico_regions.sh) for easy debugging:
```bash
bash scripts/debug_hico_regions.sh 3  # Debug first 3 images
```

## How to Debug

### Step 1: Run Debug Script

```bash
# Debug first 3 images with verbose output
bash scripts/debug_hico_regions.sh 3

# Or use environment variables for custom paths
MODEL_PATH=/path/to/model bash scripts/debug_hico_regions.sh 5
```

### Step 2: Examine Console Output

Look for the new debug sections:

```
[Image 3/10] HICO_test2015_00000003.jpg
  Categories in image: ['person', 'boat']

  🔍 DEBUG - Token Analysis for [person]:
     Generated text: <r#9><r#9></s>
     Box tokens found: [32009, 32009]
     Mapped to region indices: [9, 9]
     Total available regions: 10
     ⚠️  DUPLICATE REGIONS DETECTED: [9]
        Region R9 selected 2 times

  [person] ✓ SUCCESS
    Selected regions: R9, R9
    Box 1 (R9): [230.3, 149.1, 209.5, 141.1]
    Box 2 (R9): [230.3, 149.1, 209.5, 141.1]
```

This tells you:
1. **What the model generated**: `<r#9><r#9></s>` (duplicate tokens)
2. **Why the same box appears twice**: Model generated duplicate tokens
3. **Not an IoU bug**: The token-to-index mapping is working correctly

### Step 3: Check Visualizations

Open the visualization images in `results/hico_ground_debug/visualizations_*/`:

Each image has:
- **Left panel**: All 10 detected regions (R0-R9), with selected regions in gold
- **Middle panel**: Category predictions with IoU colors
- **Right panel**: Ground truth boxes
- **Bottom legend**: Shows generated tokens with `[DUPLICATES: R9×2]` in RED

### Step 4: Analyze the Issue

The visualizations will show you:

1. ✅ **If all regions are correct**: Left panel shows all objects properly detected
2. ❌ **Duplicate selection**: Middle panel shows same box drawn twice (offset with dashed lines)
3. 🔍 **Token output**: Bottom shows exactly what tokens the model generated

## Possible Solutions

Based on the debug output, you can determine:

### Case 1: Model Generates Duplicate Tokens

**Symptom**: Debug shows `[9, 9]` in region indices

**Cause**: Model instruction finetuning issue - model is repeating the same token

**Solution**:
- Improve instruction finetuning data to teach model to output multiple unique boxes
- Add training examples with multiple instances of same category
- Consider post-processing to deduplicate boxes (using NMS or IoU threshold)

### Case 2: Wrong Regions Selected

**Symptom**: Model generates `[9, 5]` but boxes don't match objects

**Cause**: Region proposer finds objects but model instruction alignment fails

**Solution**:
- Improve category-specific instruction tuning
- Check if query format matches training data

### Case 3: Not Enough Regions Generated

**Symptom**: Only 2-3 regions in left panel instead of 10

**Cause**: Region proposer (detection) not working well

**Solution**:
- Lower box_score_thres (currently 0.15)
- Improve detection pretraining

## Modified Files

1. **[groma/eval/eval_hico_ground.py](groma/eval/eval_hico_ground.py)**:
   - Lines 85-98: Updated `visualize_regions()` signature to accept `generated_text_per_category`
   - Lines 347-370: Added token debug info to visualization
   - Lines 694-710: Added console token analysis debug output
   - Lines 756-762: Enhanced box output with region indices
   - Line 825: Pass `generated_text_per_category` to visualization

2. **[scripts/debug_hico_regions.sh](scripts/debug_hico_regions.sh)**:
   - New debug script for easy testing

## Usage Examples

```bash
# Quick debug (3 images)
bash scripts/debug_hico_regions.sh 3

# Deeper investigation (10 images)
bash scripts/debug_hico_regions.sh 10

# Custom model
MODEL_PATH=/path/to/checkpoint bash scripts/debug_hico_regions.sh 5

# Check specific image
# Look in results/hico_ground_debug/visualizations_*/ for per-image visualizations
```

## Expected Output

After running the debug script, you should see:

1. **Console**: Detailed token analysis showing exactly which regions were selected
2. **Visualizations**: Side-by-side comparison of all regions vs selected regions
3. **Log file**: Complete debug session saved for analysis
4. **Diagnosis**: Clear indication of whether issue is:
   - Instruction alignment (model generates wrong tokens)
   - Region proposer (not enough good regions)
   - Duplicate generation (model repeats same token)

## Next Steps

1. Run the debug script on a few problematic images
2. Check if duplicates appear in the token analysis
3. Examine visualizations to see if correct regions exist but aren't selected
4. Based on findings, choose appropriate solution (retraining, post-processing, etc.)
