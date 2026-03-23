#!/usr/bin/env python3
"""
Stage1: Full SFT on all reasoning data (single A800 80GB, GPU 7).

Changes from RTX 5090 version:
  - Full fine-tune (no LoRA) — A800 80GB has enough memory
  - All 13,023 reasoning samples (no sampling)
  - max_length=8192 (covers 99.3% of data without truncation)
  - Single GPU: CUDA_VISIBLE_DEVICES=7
  - Only saves final checkpoint (no intermediate saves)
  - Runs inference after training to verify quality
"""
import os
import re
import sys
import json
import time
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

# Paths — edit these to match the A800 cluster layout
BASE_MODEL    = "/mdr5/project/DeepAnalyze_refactor_exp1/models/DeepSeek-R1-0528-Qwen3-8B"
DATA_DIR      = "/mdr5/project/DeepAnalyze_refactor_exp1/DataScience-Instruct-500K"
EXP_DIR       = Path("/mdr5/project/DeepAnalyze_refactor_exp1/experiment")
WORKSPACE_DIR = "/mdr5/project/DeepAnalyze_refactor_exp1/example/analysis_on_student_loan/data"

GPU_ID        = "7"

# Training hyperparams
MAX_LENGTH    = 8192     # covers 99.3% of data (P99=6436, max outlier=144k)
BATCH_SIZE    = 2        # per-device batch size (A800 80GB can handle this)
GRAD_ACCUM    = 8        # effective batch = 16
LR            = 1e-5     # lower LR for full fine-tune (vs 5e-5 for LoRA)
EPOCHS        = 1
WARMUP_RATIO  = 0.05

# Derived paths
TRAIN_DATA    = str(EXP_DIR / "data" / "stage1" / "train.jsonl")
OUTPUT_DIR    = str(EXP_DIR / "checkpoints" / "stage1_sft")
LOG_FILE      = EXP_DIR / "checkpoints" / "stage1_sft" / "train.log"

# ── Step 0: Prepare data (all reasoning, no sampling) ────────────────────────

def prepare_all_reasoning_data():
    """Load ALL reasoning JSON files and convert to ms-swift format."""
    reasoning_dir = Path(DATA_DIR) / "reasoning"
    files = sorted(reasoning_dir.glob("*.json"))
    if not files:
        print(f"ERROR: No JSON files found in {reasoning_dir}")
        sys.exit(1)

    all_items = []
    for f in files:
        with open(f) as fh:
            items = json.load(fh)
        print(f"  Loaded {len(items):>5} from {f.name}")
        all_items.extend(items)

    print(f"  Total: {len(all_items)} reasoning samples")

    # Filter out extreme outliers (>8192 tokens) to avoid wasting GPU time
    filtered = [item for item in all_items if item.get("total_tokens", 0) <= MAX_LENGTH]
    dropped = len(all_items) - len(filtered)
    if dropped > 0:
        print(f"  Dropped {dropped} samples with >{MAX_LENGTH} tokens ({dropped/len(all_items)*100:.1f}%)")
    print(f"  Keeping {len(filtered)} samples for training")

    # Convert to ms-swift format
    swift_data = [{"messages": item["messages"]} for item in filtered]

    out_path = Path(TRAIN_DATA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in swift_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved → {out_path}")
    return len(swift_data)


# ── Progress monitor ──────────────────────────────────────────────────────────

class TrainingMonitor:
    def __init__(self, total_steps):
        self.start_time = time.time()
        self.last_loss = None
        self.last_step = 0
        self.total_steps = total_steps
        self.bar_width = 40

    def parse_line(self, line):
        step_m = re.search(r'"step":\s*(\d+)', line)
        loss_m = re.search(r'"loss":\s*([\d.]+)', line)
        if step_m:
            return int(step_m.group(1)), float(loss_m.group(1)) if loss_m else self.last_loss
        return None, None

    def update(self, line):
        step, loss = self.parse_line(line)
        if step and step != self.last_step:
            self.last_step = step
            self.last_loss = loss
            elapsed = time.time() - self.start_time
            pct = min(step / self.total_steps, 1.0)
            filled = int(self.bar_width * pct)
            bar = "=" * filled + ">" + "." * (self.bar_width - filled - 1)
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            loss_str = f"loss={loss:.4f}" if loss else ""
            print(f"\r[{elapsed/60:.0f}m] [{bar}] {step}/{self.total_steps} ({pct*100:.1f}%) ETA {eta/60:.0f}m {loss_str}",
                  end="", flush=True)
            if step % 50 == 0:
                print()  # newline every 50 steps


# ── Step 1: Train ─────────────────────────────────────────────────────────────

def train(n_samples):
    total_steps = n_samples // (BATCH_SIZE * GRAD_ACCUM)
    # Only save at the end
    save_steps = total_steps + 1  # never triggers mid-training

    cmd = [
        "swift", "sft",
        "--model",                       BASE_MODEL,
        "--train_type",                  "full",
        "--dataset",                     TRAIN_DATA,
        "--torch_dtype",                 "bfloat16",
        "--num_train_epochs",            str(EPOCHS),
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRAD_ACCUM),
        "--learning_rate",               str(LR),
        "--max_length",                  str(MAX_LENGTH),
        "--truncation_strategy",         "right",
        "--gradient_checkpointing",      "true",
        "--warmup_ratio",                str(WARMUP_RATIO),
        "--save_steps",                  str(save_steps),
        "--save_total_limit",            "1",
        "--logging_steps",               "1",
        "--output_dir",                  OUTPUT_DIR,
        "--attn_impl",                   "flash_attn",
        "--packing",                     "false",
        "--dataloader_num_workers",      "4",
        "--model_type",                  "qwen3",
    ]

    print(f"\nTraining config:")
    print(f"  Model:          {BASE_MODEL}")
    print(f"  Train type:     full (no LoRA)")
    print(f"  Samples:        {n_samples}")
    print(f"  Max length:     {MAX_LENGTH}")
    print(f"  Batch (eff):    {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Total steps:    {total_steps}")
    print(f"  LR:             {LR}")
    print(f"  GPU:            cuda:{GPU_ID}")
    print(f"  Output:         {OUTPUT_DIR}")
    print()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    monitor = TrainingMonitor(total_steps)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["QT_QPA_PLATFORM"] = "offscreen"

    start = time.time()
    with open(LOG_FILE, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if any(kw in line for kw in ["Error", "error", "Traceback", "WARNING", "epoch", "train_loss"]):
                print(f"\n[LOG] {line.rstrip()}")
            monitor.update(line)
        proc.wait()

    elapsed = time.time() - start
    print(f"\n\n{'='*60}")
    if proc.returncode == 0:
        print(f" SFT COMPLETE in {elapsed/60:.1f} min")
    else:
        print(f" SFT FAILED (rc={proc.returncode}) after {elapsed/60:.1f} min")
        print(f" See log: {LOG_FILE}")
        sys.exit(1)
    print("="*60)


# ── Step 2: Find checkpoint and export ────────────────────────────────────────

def find_checkpoint():
    """ms-swift saves to output_dir/v{N}-{timestamp}/checkpoint-{step}/."""
    ckpt_dirs = sorted(
        Path(OUTPUT_DIR).rglob("checkpoint-*"),
        key=lambda p: int(re.search(r'checkpoint-(\d+)', str(p)).group(1))
            if re.search(r'checkpoint-(\d+)', str(p)) else 0,
        reverse=True,
    )
    if ckpt_dirs:
        print(f"  Found checkpoint: {ckpt_dirs[0]}")
        return str(ckpt_dirs[0])
    # Fallback: ms-swift may save directly to output_dir with config.json
    if (Path(OUTPUT_DIR) / "config.json").exists():
        return OUTPUT_DIR
    print("  WARNING: No checkpoint found!")
    return None


# ── Step 3: Inference ─────────────────────────────────────────────────────────

def run_inference(model_path):
    """Run vLLM inference on student loan task to verify quality."""
    print(f"\n{'='*60}")
    print(f" Running inference with trained model")
    print(f" Model: {model_path}")
    print(f"{'='*60}\n")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Build prompt (same as infer.py)
    ws = Path(WORKSPACE_DIR)
    csv_files = sorted(ws.glob("*.csv"))
    ws_lines = [f"=== Workspace: {ws} ===", f"Files: {[f.name for f in csv_files]}", ""]
    for csv_f in csv_files[:5]:
        try:
            content = csv_f.read_text(errors="replace").splitlines()
            ws_lines.append(f"--- {csv_f.name} (first 3 rows) ---")
            ws_lines.append("\n".join(content[:4]))
            ws_lines.append("")
        except Exception:
            pass
    ws_summary = "\n".join(ws_lines)

    system_msg = (
        "You are an expert data scientist. You have access to a data workspace. "
        "Analyze the data and generate a comprehensive data science report."
    )
    user_msg = (
        f"{ws_summary}\n\n"
        "Task: Generate a comprehensive data science report on the student loan dataset. "
        "Include: data overview, key statistics, insights, and conclusions."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=7000,
        stop=["</Answer>", "<|im_end|>"],
    )

    t0 = time.time()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.time() - t0
    response = outputs[0].outputs[0].text
    n_tokens = len(outputs[0].outputs[0].token_ids)

    print(f"  Generated {n_tokens} tokens in {elapsed:.1f}s")

    # Save
    out_dir = EXP_DIR / "sft_model_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "stage": "sft_full",
        "model_path": model_path,
        "timestamp": ts,
        "tokens_generated": n_tokens,
        "elapsed_s": round(elapsed, 2),
        "response": response,
    }
    (out_dir / f"sft_result_{ts}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out_dir / f"sft_response_{ts}.txt").write_text(response)

    print(f"\n{'='*60}")
    print("RESPONSE PREVIEW (first 1000 chars):")
    print("="*60)
    print(response[:1000])
    print(f"\n... ({len(response)} chars total)")
    print("="*60)

    del llm
    return response


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print(f" Stage1 Full SFT — A800 80GB (GPU {GPU_ID})")
    print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 0: Prepare data
    print("\n[Step 0] Preparing all reasoning data...")
    n_samples = prepare_all_reasoning_data()

    # Step 1: Train
    print(f"\n[Step 1] Training full SFT on {n_samples} samples...")
    train(n_samples)

    # Step 2: Find checkpoint
    print("\n[Step 2] Locating checkpoint...")
    ckpt = find_checkpoint()
    if not ckpt:
        print("ERROR: No checkpoint found after training.")
        sys.exit(1)

    # Step 3: Inference
    print(f"\n[Step 3] Running inference to verify quality...")
    run_inference(ckpt)

    print(f"\n{'='*60}")
    print(f" ALL DONE")
    print(f" Checkpoint: {ckpt}")
    print(f" End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
