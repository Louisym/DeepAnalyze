#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM inference.
ms-swift export command handles this cleanly.

Usage:
    python experiment/merge_lora.py --adapter_path checkpoints/stage1_sft/checkpoint-125 \
                                    --output_path checkpoints/stage1_sft_merged
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Load .env
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ADDVOCAB_MODEL = os.environ.get("BASE_MODEL_ADDVOCAB_PATH", "")
SWIFT_DIR = Path("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/ms-swift")


def find_latest_checkpoint(base_dir):
    base = Path(base_dir)
    # ms-swift saves to: base/v{N}-{timestamp}/checkpoint-{step}/adapter_config.json
    adapters = list(base.rglob("adapter_config.json"))
    if adapters:
        # Pick the one with the highest checkpoint step number
        def step_num(p):
            try:
                return int(p.parent.name.split("-")[-1])
            except Exception:
                return 0
        return str(sorted(adapters, key=step_num)[-1].parent)
    # Fallback: direct checkpoint-N subdirs
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        return str(ckpts[-1])
    if (base / "adapter_config.json").exists() or (base / "config.json").exists():
        return str(base)
    return str(base)


def merge(adapter_path: str, output_path: str):
    adapter = Path(adapter_path)
    output  = Path(output_path)

    # Check if adapter or full model
    is_lora = (adapter / "adapter_config.json").exists()

    if not is_lora:
        print(f"  Not a LoRA adapter (no adapter_config.json found in {adapter})")
        print(f"  Assuming full model — using directly: {adapter}")
        return str(adapter)

    print(f"  Merging LoRA adapter: {adapter}")
    print(f"  Output: {output}")

    if output.exists() and (output / "config.json").exists():
        print(f"  Merged model already exists, skipping: {output}")
        return str(output)

    # swift export fails if output_dir exists at all — remove it first
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "swift", "export",
        "--adapters", str(adapter),
        "--merge_lora", "true",
        "--output_dir", str(output),
    ]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # merge on CPU

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env=env,
        cwd=str(SWIFT_DIR),
    )

    if result.returncode != 0:
        print(f"  ERROR: merge failed (returncode={result.returncode})")
        sys.exit(1)

    print(f"  Merge complete in {time.time()-t0:.1f}s → {output}")
    return str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    adapter = find_latest_checkpoint(args.adapter_path)
    result = merge(adapter, args.output_path)
    print(f"Merged model path: {result}")


if __name__ == "__main__":
    main()
