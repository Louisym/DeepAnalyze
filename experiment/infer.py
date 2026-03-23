#!/usr/bin/env python3
"""
DeepAnalyze inference script.
Runs vLLM inference on the student loan test task.

Usage:
    python experiment/infer.py --stage base
    python experiment/infer.py --stage sft    --model_path /path/to/sft/checkpoint
    python experiment/infer.py --stage cold   --model_path /path/to/cold/checkpoint
    python experiment/infer.py --stage rl     --model_path /path/to/rl/checkpoint
"""
import os
import sys
import json
import argparse
import time
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

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "")
EXPERIMENT_DIR = os.environ.get("EXPERIMENT_DIR", "")
BASE_MODEL_ADDVOCAB = os.environ.get("BASE_MODEL_ADDVOCAB_PATH", "")

STAGE_OUTPUT_MAP = {
    "base":  "base_model_output",
    "sft":   "sft_model_output",
    "cold":  "cold_start_output",
    "rl":    "rl_output",
}

# ── Build prompt ──────────────────────────────────────────────────────────────

def build_workspace_summary(workspace_dir: str) -> str:
    """List CSV files in workspace and show first few lines of each."""
    lines = []
    ws = Path(workspace_dir)
    csv_files = sorted(ws.glob("*.csv"))
    lines.append(f"=== Workspace: {ws} ===")
    lines.append(f"Files: {[f.name for f in csv_files]}")
    lines.append("")
    for csv_f in csv_files[:5]:  # show first 5 tables
        try:
            content = csv_f.read_text(errors="replace").splitlines()
            lines.append(f"--- {csv_f.name} (first 3 rows) ---")
            lines.append("\n".join(content[:4]))
            lines.append("")
        except Exception as e:
            lines.append(f"--- {csv_f.name}: error reading: {e} ---")
    return "\n".join(lines)


def build_prompt(workspace_dir: str, stage: str = "") -> str:
    ws_summary = build_workspace_summary(workspace_dir)
    if stage == "rl":
        # RL model was trained with DeepAnalyze tag format
        system_msg = (
            "You are an expert data scientist. "
            "Always respond using the following format:\n"
            "<Analyze>\n[your analysis and code]\n</Analyze>\n"
            "<Answer>\n[your final answer]\n</Answer>"
        )
    else:
        system_msg = (
            "You are an expert data scientist. You have access to a data workspace. "
            "Analyze the data and generate a comprehensive data science report."
        )
    user_msg = (
        f"{ws_summary}\n\n"
        "Task: Generate a comprehensive data science report on the student loan dataset. "
        "Include: data overview, key statistics, insights, and conclusions."
    )
    return system_msg, user_msg


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model_path: str, stage: str, output_dir: Path):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f" Stage: {stage.upper()}")
    print(f" Model: {model_path}")
    print(f" Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # Build prompt
    print("[2/4] Building prompt...")
    system_msg, user_msg = build_prompt(WORKSPACE_DIR, stage=stage)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # For RL stage: force <Analyze> prefix (matches training distribution)
    if stage == "rl":
        prompt_text = prompt_text + "<Analyze>"
    print(f"  Prompt length: {len(prompt_text)} chars")

    # Load model with vLLM
    print("[3/4] Loading model with vLLM (this takes a few minutes)...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Generate
    print("[4/4] Generating response...")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=7000,
        stop=["</Answer>", "<|im_end|>"],
    )
    t1 = time.time()
    outputs = llm.generate([prompt_text], sampling_params)
    elapsed = time.time() - t1

    response = outputs[0].outputs[0].text
    tokens_generated = len(outputs[0].outputs[0].token_ids)
    print(f"  Generated {tokens_generated} tokens in {elapsed:.1f}s ({tokens_generated/elapsed:.1f} tok/s)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "stage": stage,
        "model_path": model_path,
        "timestamp": timestamp,
        "prompt_chars": len(prompt_text),
        "tokens_generated": tokens_generated,
        "elapsed_s": round(elapsed, 2),
        "response": response,
    }

    out_json = output_dir / f"{stage}_result_{timestamp}.json"
    out_txt  = output_dir / f"{stage}_response_{timestamp}.txt"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    out_txt.write_text(response)

    print(f"\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_txt}")
    print(f"\n{'─'*60}")
    print("RESPONSE PREVIEW (first 800 chars):")
    print("─"*60)
    print(response[:800])
    print("─"*60)

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["base", "sft", "cold", "rl"],
                        help="Which experiment stage")
    parser.add_argument("--model_path", default=None,
                        help="Model path (defaults to addvocab base for 'base' stage)")
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    elif args.stage == "base":
        model_path = BASE_MODEL_ADDVOCAB
    else:
        print(f"ERROR: --model_path required for stage '{args.stage}'")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"ERROR: model path not found: {model_path}")
        sys.exit(1)

    output_subdir = STAGE_OUTPUT_MAP[args.stage]
    output_dir = Path(EXPERIMENT_DIR) / output_subdir

    run_inference(model_path, args.stage, output_dir)


if __name__ == "__main__":
    main()
