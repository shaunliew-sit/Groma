# Groma Qwen3VL - Referring Task Implementation

This document lists all files created/modified for the Groma Qwen3VL referring task implementation.
All files are marked with `[GROMA-QWEN]` in their headers for easy identification.

## Quick Find

```bash
# Find all Groma Qwen files
grep -rl "\[GROMA-QWEN\]" . --include="*.py" --include="*.sh"
```

---

## Architecture

### Model Core

| File | Purpose |
|------|---------|
| `groma/model/groma_qwen.py` | **Main Model** - GromaQwenModel extending Qwen3VL with DINOv2 region encoder, VL Bridge, and special token handling |
| `groma/model/roi_align.py` | **Region Feature Extraction** - Multi-level ROI Align with feature fusion from DINOv2 |

### Training

| File | Purpose |
|------|---------|
| `groma/train/train_qwen.py` | Training script with Stage 2/3 support, special token handling |
| `groma/train/train_qwen_mem.py` | Memory-efficient training entry point (disables bitsandbytes) |
| `scripts/vl_pretrain_hoi_combined.sh` | **Stage 2** - VL Alignment Pretraining (trains VL Bridge + Region Encoder) |
| `scripts/vl_finetune_referring_only.sh` | **Stage 3** - Referring-only fine-tuning with new Qwen3VL-native instructions |

### Dataset Configs

| File | Purpose |
|------|---------|
| `groma/data/configs/vl_pretrain_hoi_combined.py` | Stage 2 dataset config (HICO + SWIG combined) |
| `groma/data/configs/vl_finetune_hoi_combined_qwen.py` | Stage 3 combined dataset config (grounding + referring) |
| `groma/data/configs/vl_finetune_referring_only_qwen.py` | Stage 3 **referring-only** dataset config |

### Evaluation

| File | Purpose |
|------|---------|
| `groma/eval/eval_hico_action_referring_groma_qwen.py` | HICO-DET evaluation with METEOR, CIDEr, BLEU, ROUGE-L |
| `groma/eval/eval_swig_action_referring_groma_qwen.py` | SWIG-HOI evaluation with METEOR, CIDEr, BLEU, ROUGE-L |
| `scripts/run_hico_action_referring_groma_qwen_eval.sh` | Shell script to run HICO evaluation |
| `scripts/run_swig_action_referring_groma_qwen_eval.sh` | Shell script to run SWIG evaluation |

### Testing / Debugging

| File | Purpose |
|------|---------|
| `scripts/test_groma_qwen_referring.py` | Interactive test script for referring task capabilities |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GromaQwenModel                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐ │
│  │   DINOv2     │    │  Region Encoder  │    │      VL Bridge         │ │
│  │   (Frozen)   │───▶│  (MLVLROIQuery)  │───▶│  (img_txt_bridge)      │ │
│  │  1024-dim    │    │   ROI Align      │    │  4096 → text_embed_dim │ │
│  └──────────────┘    └──────────────────┘    └───────────┬────────────┘ │
│                                                          │              │
│  ┌──────────────────────────────────────────────────────▼────────────┐ │
│  │                      Qwen3VL-8B-Instruct                          │ │
│  │  (Frozen in Stage 2, Trained in Stage 3)                          │ │
│  │                                                                    │ │
│  │  Embedding Hook: Injects region features at <refer_feat> tokens   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  New Token Embeddings: <region>, <refer_feat>, <r0>...<r99>, etc │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

### Stage 2: VL Alignment Pretraining

```bash
bash scripts/vl_pretrain_hoi_combined.sh \
    checkpoints/Qwen3-VL-8B-Instruct \
    checkpoints/dinov2-large \
    checkpoints/groma-qwen-stage2-hoi-qwen3vl \
    "0,6,7"
```

- **Frozen**: LLM (Qwen3VL), DINOv2
- **Trained**: VL Bridge, Region Encoder

### Stage 3: Instruction Finetuning (Referring-Only)

```bash
bash scripts/vl_finetune_referring_only.sh \
    checkpoints/groma-qwen-stage2-hoi-qwen3vl \
    checkpoints/groma-qwen-stage3-referring-only \
    "0,6,7"
```

- **Frozen**: DINOv2
- **Trained**: LLM, VL Bridge, Region Encoder, New Token Embeddings

---

## Evaluation

### HICO-DET Action Referring

```bash
# Quick test (10 samples)
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0

# Full evaluation
bash scripts/run_hico_action_referring_groma_qwen_eval.sh 0
```

### SWIG-HOI Action Referring

```bash
# Quick test (10 samples)
VERBOSE=1 MAX_IMAGES=10 bash scripts/run_swig_action_referring_groma_qwen_eval.sh 0

# Full evaluation
bash scripts/run_swig_action_referring_groma_qwen_eval.sh 0
```

---

## Interactive Testing

```bash
python scripts/test_groma_qwen_referring.py \
    --model-name checkpoints/groma-qwen-stage3-referring-only \
    --image-file path/to/image.jpg \
    --subject-bbox "320,306,359,349" \
    --object-bbox "148,345,376,414" \
    --gpu-id 0
```

---

## Key Changes Summary

### Referring Task Fix (Task Confusion Issue)

**Problem**: Model was outputting coordinates instead of action phrases for referring tasks.

**Solution**:
1. Created new Qwen3VL-native instruction format with explicit task descriptions
2. Trained on referring-only dataset (no grounding samples)
3. Fixed `forward()` to inject features at `<refer_feat>` tokens (not just `<region>`)
4. Fixed `prepare_inputs_for_generation()` to preserve region data during generation

### dtype Compatibility (roi_align.py)

DINOv2 outputs float32, but Qwen3VL uses bfloat16. Added dtype conversions throughout:
- `MLVLFuseModule._single_shuffle()`: F.interpolate in float32, convert back
- `MLVLROIQueryModule.forward()`: dtype conversion before/after interpolation
- `MlvlRoIExtractor.forward()`: roi_layers use float32, output converted to model dtype

---

## Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `checkpoints/Qwen3-VL-8B-Instruct` | Base LLM |
| `checkpoints/dinov2-large` | DINOv2 visual encoder |
| `checkpoints/groma-qwen-stage2-hoi-qwen3vl-24-nov-fixed` | Stage 2 checkpoint |
| `checkpoints/groma-qwen-stage3-referring-only-25-nov-fixed` | Stage 3 referring-only checkpoint |

---

## Datasets

| Dataset | Path | Samples |
|---------|------|---------|
| HICO Referring Only | `groma_data/benchmarks/hico_referring_only_qwen3vl.json` | 93,041 |
| SWIG Referring Only | `groma_data/benchmarks/swig_referring_only_qwen3vl.json` | 60,793 |
| **Total** | | **153,834** |

