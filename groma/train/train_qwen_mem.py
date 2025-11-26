# Memory efficient training script for GromaQwenModel
# Wrapper for train_qwen.py

import sys
# Hack to disable broken bitsandbytes
sys.modules['bitsandbytes'] = None

# from groma.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

from groma.train.train_qwen import train

if __name__ == "__main__":
    train()
