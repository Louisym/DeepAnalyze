#!/usr/bin/env python3
"""Merge LoRA adapter into base model using PEFT API (CPU merge, no swift export)."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage1"

CONFIGS = {
    "stage1": {
        "adapter": "/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/checkpoints/stage1_sft/v0-20260320-155443/checkpoint-125",
        "base":    "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B-addvocab",
        "output":  "/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/checkpoints/stage1_sft_merged",
    },
    "stage2": {
        "adapter": None,  # will be found at runtime
        "base":    "/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/checkpoints/stage1_sft_merged",
        "output":  "/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/checkpoints/stage2_cold_merged",
    },
}

cfg = CONFIGS[STAGE]

# For stage2, find adapter dynamically
if cfg["adapter"] is None:
    from pathlib import Path
    adapters = sorted(Path("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/checkpoints/stage2_cold").rglob("adapter_config.json"),
                      key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0)
    if not adapters:
        print("ERROR: no stage2 adapter found"); sys.exit(1)
    cfg["adapter"] = str(adapters[-1].parent)

print(f"[merge:{STAGE}] adapter  = {cfg['adapter']}")
print(f"[merge:{STAGE}] base     = {cfg['base']}")
print(f"[merge:{STAGE}] output   = {cfg['output']}")

if os.path.exists(cfg["output"]) and os.path.exists(os.path.join(cfg["output"], "config.json")):
    print("[merge] Already exists, skipping."); sys.exit(0)

import shutil
if os.path.exists(cfg["output"]):
    shutil.rmtree(cfg["output"])

t0 = time.time()
print("[1/4] Loading base model on CPU...")
model = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
print(f"  loaded in {time.time()-t0:.0f}s")

print("[2/4] Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(cfg["base"])

print("[3/4] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, cfg["adapter"])

print("[4/4] Merging and saving...")
model = model.merge_and_unload()
os.makedirs(cfg["output"], exist_ok=True)
model.save_pretrained(cfg["output"])
tokenizer.save_pretrained(cfg["output"])

print(f"\nDone in {(time.time()-t0)/60:.1f}min → {cfg['output']}")
