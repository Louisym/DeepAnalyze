#!/usr/bin/env python3
"""
Stage2: Cold-start SFT on interaction (multi-turn) data with LoRA.
Trains from the Stage1 SFT LoRA checkpoint.

Adapted from scripts/multi_coldstart.sh:
  - full fine-tune → LoRA (32GB GPU)
  - max_length 32768 → 16384 (memory constraint)
  - batch_size adjusted for single GPU
"""
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Load .env
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EXP_DIR   = Path(os.environ.get("EXPERIMENT_DIR", ""))
SWIFT_DIR = Path("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/ms-swift")

TRAIN_DATA  = str(EXP_DIR / "data" / "stage2" / "train.jsonl")
STAGE1_CKPT = str(EXP_DIR / "checkpoints" / "stage1_sft_merged")
OUTPUT_DIR  = str(EXP_DIR / "checkpoints" / "stage2_cold")
LOG_FILE    = Path(OUTPUT_DIR) / "train.log"


def find_latest_checkpoint(base_dir):
    """Find the latest checkpoint dir or adapter file."""
    base = Path(base_dir)
    # ms-swift saves checkpoints as checkpoint-{step}/
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        return str(ckpts[-1])
    return base_dir


class TrainingMonitor:
    def __init__(self, total_steps_estimate=None):
        self.start_time = time.time()
        self.last_loss = None
        self.last_step = 0
        self.total_steps = total_steps_estimate
        self.bar_width = 40

    def parse_line(self, line):
        step_m = re.search(r'"step":\s*(\d+)', line)
        loss_m = re.search(r'"loss":\s*([\d.]+)', line)
        if step_m:
            step = int(step_m.group(1))
            loss = float(loss_m.group(1)) if loss_m else self.last_loss
            return step, loss
        return None, None

    def update(self, line):
        step, loss = self.parse_line(line)
        if step and step != self.last_step:
            self.last_step = step
            self.last_loss = loss
            elapsed = time.time() - self.start_time
            if self.total_steps and self.total_steps > 0:
                pct = min(step / self.total_steps, 1.0)
                filled = int(self.bar_width * pct)
                bar = "█" * filled + "░" * (self.bar_width - filled)
                eta = (elapsed / pct * (1 - pct)) if pct > 0 else 0
                eta_str = f"ETA {eta/60:.0f}m"
                bar_str = f"[{bar}] {step}/{self.total_steps} ({pct*100:.1f}%) {eta_str}"
            else:
                bar_str = f"Step {step}"
            loss_str = f"loss={loss:.4f}" if loss else ""
            ts = f"[{elapsed/60:.1f}m]"
            print(f"\r{ts} {bar_str} {loss_str}", end="", flush=True)
            if step % 10 == 0:
                print()


def main():
    # Find Stage1 checkpoint to use as base
    stage1_base = find_latest_checkpoint(STAGE1_CKPT)
    print("="*60)
    print(f" Stage2 Cold-Start SFT Training")
    print(f" Base model (Stage1): {stage1_base}")
    print(f" Data:   {TRAIN_DATA}")
    print(f" Output: {OUTPUT_DIR}")
    print(f" Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not Path(stage1_base).exists():
        print(f"ERROR: Stage1 checkpoint not found: {stage1_base}")
        sys.exit(1)
    if not Path(TRAIN_DATA).exists():
        print(f"ERROR: Train data not found: {TRAIN_DATA}")
        sys.exit(1)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Stage2: longer sequences (interaction), smaller batch, lower lr
    # 1200 samples, batch=1, grad_accum=16 → effective batch=16, ~75 steps/epoch
    cmd = [
        "swift", "sft",
        "--model",                       stage1_base,
        "--train_type",                  "lora",
        "--lora_rank",                   "16",
        "--lora_alpha",                  "32",
        "--dataset",                     TRAIN_DATA,
        "--torch_dtype",                 "bfloat16",
        "--num_train_epochs",            "1",
        "--max_steps",                   "30",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--learning_rate",               "5e-6",
        "--max_length",                  "4096",
        "--truncation_strategy",         "right",
        "--gradient_checkpointing",      "true",
        "--warmup_ratio",                "0.05",
        "--eval_steps",                  "30",
        "--save_steps",                  "30",
        "--logging_steps",               "1",
        "--save_total_limit",            "2",
        "--output_dir",                  OUTPUT_DIR,
        "--attn_impl",                   "flash_attn",
        "--packing",                     "false",
        "--dataloader_num_workers",      "4",
        "--model_type",                  "qwen3",
    ]

    print(f"\nCommand: {' '.join(cmd[:6])} ...")
    print(f"Log:     {LOG_FILE}\n")

    # ~75 steps estimated
    monitor = TrainingMonitor(total_steps_estimate=75)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["QT_QPA_PLATFORM"] = "offscreen"

    start = time.time()
    with open(LOG_FILE, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(SWIFT_DIR),
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if any(kw in line for kw in ["Error", "error", "Traceback", "epoch", "train_loss"]):
                print(f"\n[LOG] {line.rstrip()}")
            monitor.update(line)
        proc.wait()

    elapsed = time.time() - start
    print(f"\n\n{'='*60}")
    if proc.returncode == 0:
        print(f" Stage2 Cold-Start SFT COMPLETE in {elapsed/60:.1f} min")
        print(f" Checkpoint: {OUTPUT_DIR}")
    else:
        print(f" Stage2 FAILED (returncode={proc.returncode}) after {elapsed/60:.1f} min")
        print(f" See log: {LOG_FILE}")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
