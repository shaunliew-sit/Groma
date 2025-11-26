# [GROMA-QWEN] Memory efficient training script for GromaQwenModel
# Part of: Groma Qwen3VL Referring Task Implementation
# Wrapper for train_qwen.py
#
# Purpose:
# --------
# Entry point for training that disables problematic bitsandbytes import.
# Use this instead of train_qwen.py directly when running training scripts.
#
# Usage:
# ------
# Called by: scripts/vl_pretrain_hoi_combined.sh (Stage 2)
#            scripts/vl_finetune_referring_only.sh (Stage 3)
#

import sys
# Hack to disable broken bitsandbytes
sys.modules['bitsandbytes'] = None

# from groma.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

from groma.train.train_qwen import train

if __name__ == "__main__":
    train()
