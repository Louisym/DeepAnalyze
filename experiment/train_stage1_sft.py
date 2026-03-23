#!/usr/bin/env python3
"""
Stage1: Reasoning SFT training with LoRA (single GPU, 5090 32GB).
Uses ms-swift via subprocess with real-time tqdm progress monitoring.

Adapted from scripts/single.sh:
  - full fine-tune → LoRA (GPU memory: 32GB constraint)
  - 8-GPU → 1 GPU
  - deepspeed zero3 → disabled
  - 2000 sampled reasoning examples
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

ADDVOCAB_MODEL = os.environ.get("BASE_MODEL_ADDVOCAB_PATH", "")
EXP_DIR        = Path(os.environ.get("EXPERIMENT_DIR", ""))
SWIFT_DIR      = Path("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/ms-swift")

TRAIN_DATA     = str(EXP_DIR / "data" / "stage1" / "train.jsonl")
OUTPUT_DIR     = str(EXP_DIR / "checkpoints" / "stage1_sft")
LOG_FILE       = EXP_DIR / "checkpoints" / "stage1_sft" / "train.log"


# ── Progress monitor ──────────────────────────────────────────────────────────

class TrainingMonitor:
    def __init__(self, total_steps_estimate=None):
        self.start_time = time.time()
        self.last_loss = None
        self.last_step = 0
        self.total_steps = total_steps_estimate
        self.bar_width = 40

    def parse_line(self, line):
        """Extract step/loss info from ms-swift log lines."""
        # Pattern: {"loss": 2.34, "grad_norm": ..., "step": 10, ...}
        step_m = re.search(r'"step":\s*(\d+)', line)
        loss_m = re.search(r'"loss":\s*([\d.]+)', line)

        if step_m:
            step = int(step_m.group(1))
            loss = float(loss_m.group(1)) if loss_m else self.last_loss
            return step, loss
        return None, None

    def render_bar(self, step):
        if not self.total_steps:
            return f"Step {step}"
        pct = min(step / self.total_steps, 1.0)
        filled = int(self.bar_width * pct)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        elapsed = time.time() - self.start_time
        if pct > 0:
            eta = elapsed / pct * (1 - pct)
            eta_str = f"ETA {eta/60:.0f}m"
        else:
            eta_str = "ETA ?"
        return f"[{bar}] {step}/{self.total_steps} ({pct*100:.1f}%) {eta_str}"

    def update(self, line):
        step, loss = self.parse_line(line)
        if step and step != self.last_step:
            self.last_step = step
            self.last_loss = loss
            elapsed = time.time() - self.start_time
            bar = self.render_bar(step)
            loss_str = f"loss={loss:.4f}" if loss else ""
            ts = f"[{elapsed/60:.1f}m]"
            print(f"\r{ts} {bar} {loss_str}", end="", flush=True)
            if step % 10 == 0:
                print()  # newline every 10 steps


# ── Build swift command ───────────────────────────────────────────────────────

def build_swift_cmd():
    # Estimate total steps: 2000 samples / batch_size=2 * grad_accum=8 * epochs=1
    # = 2000/2/8 = 125 steps per epoch, 1 epoch = 125 steps
    return [
        "swift", "sft",
        "--model",                    ADDVOCAB_MODEL,
        "--train_type",               "lora",
        "--lora_rank",                "16",
        "--lora_alpha",               "32",
        "--dataset",                  TRAIN_DATA,
        "--torch_dtype",              "bfloat16",
        "--num_train_epochs",         "1",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--learning_rate",            "5e-5",
        "--max_length",               "1792",
        "--truncation_strategy",      "right",
        "--gradient_checkpointing",   "true",
        "--warmup_ratio",             "0.05",
        "--eval_steps",               "50",
        "--save_steps",               "50",
        "--logging_steps",            "1",
        "--save_total_limit",         "2",
        "--output_dir",               OUTPUT_DIR,
        "--attn_impl",                "flash_attn",
        "--packing",                  "false",
        "--dataloader_num_workers",   "4",
        "--model_type",               "qwen3",
    ]


def main():
    print("="*60)
    print(f" Stage1 SFT Training")
    print(f" Model: {ADDVOCAB_MODEL}")
    print(f" Data:  {TRAIN_DATA}")
    print(f" Output:{OUTPUT_DIR}")
    print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Verify inputs
    if not Path(ADDVOCAB_MODEL).exists():
        print(f"ERROR: model not found: {ADDVOCAB_MODEL}")
        sys.exit(1)
    if not Path(TRAIN_DATA).exists():
        print(f"ERROR: train data not found: {TRAIN_DATA}")
        sys.exit(1)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = build_swift_cmd()
    print(f"\nCommand: {' '.join(cmd[:6])} ...")
    print(f"Log:     {LOG_FILE}\n")

    # Estimated steps for progress bar
    # 2000 samples, batch=2, grad_accum=8 → effective batch=16, 1 epoch = 125 steps
    monitor = TrainingMonitor(total_steps_estimate=125)

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
            # Print key lines
            if any(kw in line for kw in ["Error", "error", "Traceback", "WARNING", "epoch", "train_loss"]):
                print(f"\n[LOG] {line.rstrip()}")
            monitor.update(line)

        proc.wait()

    elapsed = time.time() - start
    print(f"\n\n{'='*60}")
    if proc.returncode == 0:
        print(f" Stage1 SFT COMPLETE in {elapsed/60:.1f} min")
        print(f" Checkpoint: {OUTPUT_DIR}")
    else:
        print(f" Stage1 SFT FAILED (returncode={proc.returncode}) after {elapsed/60:.1f} min")
        print(f" See log: {LOG_FILE}")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
