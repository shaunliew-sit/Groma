# [GROMA-QWEN] Memory efficient LoRA training script for GromaQwenModel
# Part of: Groma Qwen3VL Referring Task Implementation
# Wrapper for train_qwen_lora.py
#
# Purpose:
# --------
# Entry point for LoRA training that disables problematic bitsandbytes import.
# Use this instead of train_qwen_lora.py directly when running training scripts.
#
# Usage:
# ------
# Called by: scripts/vl_finetune_referring_lora.sh
#

import sys
# Hack to disable broken bitsandbytes (same as train_qwen_mem.py)
sys.modules['bitsandbytes'] = None

from groma.train.train_qwen_lora import train

if __name__ == "__main__":
    train()

