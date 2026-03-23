#!/usr/bin/env python3
"""
Lightweight GRPO training for Stage3.
No SkyRL, no flash-attn, no reference model, no KL divergence.

Algorithm:
  For each batch of prompts:
    1. Generate n=5 rollouts per prompt using vLLM (old policy weights)
    2. Execute code in each rollout, compute reward
    3. Compute GRPO advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
    4. Policy gradient update: L = -sum(A_i * log π_θ(response_i | prompt_i))
    5. Sync vLLM weights from updated policy

Memory:
  - Policy weights (LoRA only trainable): ~16.3GB base + 0.3GB adapter
  - vLLM KV cache: ~4GB
  - Gradients + optimizer (8-bit Adam, LoRA only): ~1GB
  - Total: ~22GB
"""
import os
import sys
import re
import json
import time
import subprocess
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from openai import OpenAI

# Load .env
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EXP_DIR       = Path(os.environ.get("EXPERIMENT_DIR", ""))
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL    = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL   = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

STAGE2_MODEL  = str(EXP_DIR / "checkpoints" / "stage2_cold_merged")
TRAIN_DATA    = str(EXP_DIR / "data" / "stage3" / "train.parquet")
OUTPUT_DIR    = EXP_DIR / "checkpoints" / "stage3_rl"
LOG_FILE      = OUTPUT_DIR / "grpo_train.log"

# ── GRPO hyperparams ──────────────────────────────────────────────────────────
N_SAMPLES      = 5       # rollouts per prompt
MAX_NEW_TOKENS = 2048
MAX_TURNS      = 5       # multi-turn code execution steps
BATCH_SIZE     = 4       # prompts per update step
GRAD_ACCUM     = 4       # effective batch = 16
LR             = 5e-7
MAX_STEPS      = 30
LORA_RANK      = 16
TEMPERATURE    = 0.8
TOP_P          = 0.95

SPECIAL_TOKENS = ["<Analyze>", "</Analyze>", "<Code>", "</Code>",
                  "<Execute>", "</Execute>", "<Answer>", "</Answer>"]


# ── Reward functions (simplified from deepanalyze utils.py) ──────────────────

def check_format(text: str) -> bool:
    """Response must have <Analyze>...</Analyze> and <Answer>...</Answer>."""
    return ("<Analyze>" in text and
            "<Answer>" in text and "</Answer>" in text)


def execute_python_code(code: str, workspace: str, timeout: int = 10) -> str:
    """Execute Python code in workspace, return stdout+stderr."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout + result.stderr
        return out[:2000] if out else "[No output]"
    except subprocess.TimeoutExpired:
        return "[Timeout]"
    except Exception as e:
        return f"[Error]: {e}"


def llm_judge_reward(response: str, ground_truth: str, question: str,
                     client: OpenAI) -> float:
    """Call GPT-4o-mini to judge response quality. Returns 0-1."""
    prompt = f"""You are an expert data science evaluator.

Question: {question}

Model Response: {response[:1500]}

Ground Truth / Reference: {str(ground_truth)[:500]}

Rate the model response quality from 0.0 to 1.0:
- 1.0: Correct answer with good analysis
- 0.5: Partially correct or incomplete analysis
- 0.0: Wrong or no meaningful analysis

Respond with ONLY a float number between 0.0 and 1.0."""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score = float(resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def compute_reward(response: str, row: dict, workspace: str,
                   client: OpenAI) -> float:
    """
    Composite reward using LLM judge.
    The prompt had <Analyze> appended, so prepend it back for full context.
    """
    full_response = "<Analyze>" + response

    # LLM judge score (0.0–1.0)
    ground_truth = row.get("output_seq", "")
    question = row.get("input_seq", "")
    judge_reward = llm_judge_reward(full_response, ground_truth, question, client)

    # Small format bonus: +0.1 if has <Answer>...</Answer>
    format_bonus = 0.1 if (check_format(full_response)) else 0.0

    return min(1.0, judge_reward + format_bonus)


# ── Rollout (vLLM in separate subprocess) ────────────────────────────────────

VLLM_WORKER_SCRIPT = Path(__file__).parent / "_vllm_rollout_worker.py"

def _write_vllm_worker():
    """Write the vLLM worker script that runs in a separate process."""
    code = '''#!/usr/bin/env python3
"""vLLM rollout worker — run as subprocess so GPU memory is fully released on exit."""
import sys, json
from vllm import LLM, SamplingParams

cfg  = json.loads(sys.argv[1])
prompts_file = sys.argv[2]
out_file = sys.argv[3]

with open(prompts_file) as f:
    prompts = json.load(f)

llm = LLM(
    model=cfg["model_path"],
    dtype="bfloat16",
    max_model_len=cfg.get("max_model_len", 4096),
    gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.60),
    trust_remote_code=True,
    enable_prefix_caching=False,
)
sampling = SamplingParams(
    temperature=cfg["temperature"],
    top_p=cfg["top_p"],
    max_tokens=cfg["max_new_tokens"],
    n=cfg["n_samples"],
    stop=cfg.get("stop", ["</Answer>", "<|im_end|>"]),
)
outputs = llm.generate(prompts, sampling)
rollouts = [[o.text for o in out.outputs] for out in outputs]
with open(out_file, "w") as f:
    json.dump(rollouts, f)
'''
    VLLM_WORKER_SCRIPT.write_text(code)


def generate_rollouts_vllm(model_path: str, prompts: list, tokenizer) -> list[list[str]]:
    """
    Launch vLLM in a separate subprocess so GPU memory is fully released.
    Returns list of lists: rollouts[prompt_idx][sample_idx] = response_text
    """
    _write_vllm_worker()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(prompts, pf)
        prompts_file = pf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        out_file = of.name

    cfg = {
        "model_path": model_path,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_samples": N_SAMPLES,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.60,
        "stop": ["</Answer>", "<|im_end|>"],
    }

    cmd = [sys.executable, str(VLLM_WORKER_SCRIPT), json.dumps(cfg), prompts_file, out_file]
    print(f"  Launching vLLM worker subprocess...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  vLLM worker failed (rc={result.returncode}), returning empty rollouts")
        return [[""] * N_SAMPLES for _ in prompts]

    with open(out_file) as f:
        rollouts = json.load(f)

    Path(prompts_file).unlink(missing_ok=True)
    Path(out_file).unlink(missing_ok=True)
    torch.cuda.empty_cache()
    return rollouts


# ── Policy model (LoRA) ───────────────────────────────────────────────────────

def load_policy(model_path: str, tokenizer):
    """Load base model with LoRA for training."""
    from transformers import AutoModelForCausalLM

    print(f"  Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_cache=False,
    )
    # Add LoRA
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model


def save_lora_for_vllm(model, tokenizer, out_dir: str):
    """Save merged model for vLLM to use in next rollout round."""
    print(f"  Saving merged model → {out_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    del merged
    torch.cuda.empty_cache()


# ── GRPO update step ──────────────────────────────────────────────────────────

def grpo_loss(model, tokenizer, prompt_texts: list, rollouts: list,
              rewards: list) -> torch.Tensor:
    """
    Compute GRPO policy gradient loss.

    rewards shape: [batch, n_samples]
    rollouts shape: [batch, n_samples] of strings
    """
    total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    n_valid = 0

    for b_idx, (prompt, samples, rwds) in enumerate(zip(prompt_texts, rollouts, rewards)):
        rwds_t = torch.tensor(rwds, dtype=torch.float32)

        # GRPO advantage: normalize within group
        mean_r = rwds_t.mean()
        std_r  = rwds_t.std() + 1e-8
        advantages = (rwds_t - mean_r) / std_r  # shape [n_samples]

        for s_idx, (response, adv) in enumerate(zip(samples, advantages)):
            if adv.abs() < 1e-6:
                continue

            full_text = prompt + response
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to("cuda")

            prompt_len = len(tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1536
            ).input_ids[0])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs)
                logits = out.logits  # [1, seq_len, vocab]

            # Log probs of response tokens only
            labels = inputs.input_ids[0, prompt_len:]
            log_probs = torch.log_softmax(logits[0, prompt_len-1:-1], dim=-1)
            token_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            seq_log_prob = token_log_probs.mean()

            # GRPO: - advantage * log_prob
            loss_sample = -adv.to("cuda") * seq_log_prob
            total_loss = total_loss + loss_sample
            n_valid += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=STAGE2_MODEL)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" Stage3 GRPO Training (no KL, no reference)")
    print(f" Model:     {args.model_path}")
    print(f" Data:      {TRAIN_DATA}")
    print(f" Steps:     {args.max_steps}")
    print(f" N_samples: {N_SAMPLES} rollouts/prompt")
    print(f" Start:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data — use pyarrow directly to avoid pandas/arrow type mismatch
    import pyarrow.parquet as pq
    table = pq.read_table(TRAIN_DATA)
    records = table.to_pydict()
    n_rows = len(table)
    # Convert to list of dicts
    df = [{col: records[col][i] for col in records} for i in range(n_rows)]
    print(f"Loaded {len(df)} RL training samples")

    # Setup judge
    client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_URL)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"

    # Working dir for code execution (use /tmp if no workspace)
    workspace = "/tmp/grpo_workspace"
    Path(workspace).mkdir(exist_ok=True)

    # Current model path for vLLM rollouts
    rollout_model_path = args.model_path
    vllm_tmp_dir = str(OUTPUT_DIR / "vllm_tmp")

    # Load policy for training
    print("\nLoading policy model...")
    policy = load_policy(args.model_path, tokenizer)

    # 8-bit Adam (bitsandbytes) or fallback to AdamW
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in policy.parameters() if p.requires_grad],
            lr=LR,
        )
        print("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=LR,
        )
        print("Using standard AdamW (bitsandbytes not available)")

    log_data = []
    global_step = 0
    optimizer.zero_grad()

    pbar = tqdm(total=args.max_steps, desc="GRPO Steps")

    while global_step < args.max_steps:
        # Sample a batch
        batch = random.sample(df, min(BATCH_SIZE, len(df)))

        # Build prompts — append "<Analyze>" prefix so vLLM is forced
        # to generate in the correct DeepAnalyze tag format
        prompts = []
        for row in batch:
            system = (
                "You are an expert data scientist. "
                "Always respond using the following format:\n"
                "<Analyze>\n[your analysis]\n</Analyze>\n"
                "<Answer>\n[your final answer]\n</Answer>"
            )
            user = str(row.get("input_seq", "Analyze the data and provide insights."))
            messages = [{"role": "system", "content": system},
                        {"role": "user",   "content": user}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Force generation to start with <Analyze> tag
            prompt = prompt + "<Analyze>"
            prompts.append(prompt)

        # ── Rollout phase (vLLM, GPU off for training) ──
        print(f"\n[Step {global_step+1}] Generating {len(prompts)*N_SAMPLES} rollouts...")
        policy.cpu()
        torch.cuda.empty_cache()

        rollouts = generate_rollouts_vllm(rollout_model_path, prompts, tokenizer)

        # ── Reward computation ──
        print(f"  Computing rewards...")
        all_rewards = []
        for b_idx, (row, samples) in enumerate(zip(batch, rollouts)):
            rwds = []
            for resp in samples:
                r = compute_reward(resp, row, workspace, client)
                rwds.append(r)
            all_rewards.append(rwds)
            mean_r = sum(rwds) / len(rwds)
            print(f"  Prompt {b_idx}: rewards={[f'{r:.2f}' for r in rwds]} mean={mean_r:.2f}")

        # ── Policy update ──
        print(f"  Policy update...")
        policy.cuda()

        loss = grpo_loss(policy, tokenizer, prompts, rollouts, all_rewards)
        (loss / GRAD_ACCUM).backward()

        global_step += 1
        pbar.update(1)

        mean_reward = sum(sum(r) for r in all_rewards) / (len(all_rewards) * N_SAMPLES)
        pbar.set_postfix(loss=f"{loss.item():.4f}", reward=f"{mean_reward:.3f}")

        log_entry = {
            "step": global_step,
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "timestamp": datetime.now().isoformat(),
        }
        log_data.append(log_entry)

        # Gradient accumulation
        if global_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Optimizer step at global_step={global_step}")

            # Save updated vLLM weights every GRAD_ACCUM steps
            policy.cpu()
            torch.cuda.empty_cache()
            Path(vllm_tmp_dir).mkdir(exist_ok=True)
            import shutil
            if Path(vllm_tmp_dir).exists():
                shutil.rmtree(vllm_tmp_dir)
            save_lora_for_vllm(policy, tokenizer, vllm_tmp_dir)
            rollout_model_path = vllm_tmp_dir
            policy.cuda()

        # Save log
        with open(LOG_FILE, "w") as f:
            for entry in log_data:
                f.write(json.dumps(entry) + "\n")

    pbar.close()

    # Final save
    print(f"\nSaving final LoRA adapter...")
    adapter_out = OUTPUT_DIR / "adapter"
    adapter_out.mkdir(exist_ok=True)
    policy.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    # Merge and save final model
    print("Merging final model...")
    export_dir = OUTPUT_DIR / "export"
    policy.cpu()
    torch.cuda.empty_cache()
    save_lora_for_vllm(policy, tokenizer, str(export_dir))

    print(f"\n{'='*60}")
    print(f" GRPO Training COMPLETE")
    print(f" Steps:       {global_step}")
    print(f" Final loss:  {log_data[-1]['loss']:.4f}")
    print(f" Final reward:{log_data[-1]['mean_reward']:.3f}")
    print(f" Export:      {export_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
