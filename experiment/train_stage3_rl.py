#!/usr/bin/env python3
"""
Stage3: GRPO RL training with SkyRL.
Trains from Stage2 cold-start checkpoint.

Adapted from scripts/multi_rl.sh for single GPU (5090 32GB):
  - 8 GPUs → 1 GPU
  - train_batch_size 256 → 16 (single GPU)
  - micro_forward_batch_size 16 → 2
  - gpu_memory_utilization 0.5 → 0.4 (share GPU between policy+vLLM)
"""
import os
import re
import sys
import time
import subprocess
import argparse
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
DATA_DIR  = Path(os.environ.get("DATA_DIR", ""))
SKYRL_DIR = Path("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/SkyRL/skyrl-train")

RL_DATA   = [
    str(EXP_DIR / "data" / "stage3" / "train.parquet"),
]
OUTPUT_DIR = str(EXP_DIR / "checkpoints" / "stage3_rl")
LOG_FILE   = Path(OUTPUT_DIR) / "train.log"

# RL workspace: where code execution happens
RL_WORKSPACE = str(DATA_DIR / "RL" / "data")


class RLMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_step = 0
        self.bar_width = 40

    def update(self, line):
        # SkyRL logs: "train/reward_mean: 0.45" or "step: 10"
        step_m  = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
        reward_m = re.search(r'reward[_/]mean[:\s]+([\d.]+)', line, re.IGNORECASE)
        loss_m  = re.search(r'policy[_/]loss[:\s]+([\d.eE+\-]+)', line, re.IGNORECASE)

        if step_m:
            step = int(step_m.group(1))
            if step != self.last_step:
                self.last_step = step
                elapsed = time.time() - self.start_time
                reward  = reward_m.group(1) if reward_m else "?"
                loss    = loss_m.group(1) if loss_m else "?"
                print(f"\r[{elapsed/60:.1f}m] RL Step {step:4d} | reward={reward} | policy_loss={loss}",
                      end="", flush=True)
                if step % 5 == 0:
                    print()


def find_latest_checkpoint(base_dir):
    base = Path(base_dir)
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        return str(ckpts[-1])
    return base_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Stage2 checkpoint path (default: auto-detect)")
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_checkpoint(str(EXP_DIR / "checkpoints" / "stage2_cold"))

    print("="*60)
    print(f" Stage3 GRPO RL Training")
    print(f" Model (Stage2): {model_path}")
    print(f" RL Data: {RL_DATA}")
    print(f" Output: {OUTPUT_DIR}")
    print(f" Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not Path(model_path).exists():
        print(f"ERROR: Stage2 checkpoint not found: {model_path}")
        sys.exit(1)

    # Check if RL workspace exists
    if not Path(RL_WORKSPACE).exists():
        print(f"WARNING: RL workspace not found: {RL_WORKSPACE}")
        print("  Code execution in RL env will fail. Using /tmp as fallback.")
        workspace = "/tmp/rl_workspace"
        Path(workspace).mkdir(exist_ok=True)
    else:
        workspace = RL_WORKSPACE

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    rl_data_str = str(RL_DATA).replace("'", '"')

    cmd = [
        "python", "-m", "examples.deepanalyze.main_deepanalyze",
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.epochs=1",
        f"data.train_data={rl_data_str}",
        f"trainer.policy.model.path={model_path}",
        "trainer.placement.colocate_all=true",
        "trainer.strategy=fsdp2",
        "trainer.policy.fsdp_config.cpu_offload=true",
        "trainer.ref.fsdp_config.cpu_offload=true",
        "trainer.placement.policy_num_gpus_per_node=1",
        "trainer.placement.ref_num_gpus_per_node=1",
        "generator.num_inference_engines=1",
        "generator.inference_engine_tensor_parallel_size=1",
        "trainer.train_batch_size=16",
        "trainer.micro_forward_batch_size_per_gpu=2",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.max_prompt_length=4000",
        "generator.max_input_length=8192",
        "generator.sampling_params.max_generate_length=8192",
        "trainer.policy.optimizer_config.lr=5e-7",
        "trainer.policy_mini_batch_size=16",
        "trainer.algorithm.use_kl_loss=false",
        "generator.backend=vllm",
        "generator.run_engines_locally=true",
        "generator.weight_sync_backend=nccl",
        "generator.async_engine=true",
        "generator.batched=false",
        "generator.use_conversation_multi_turn=false",
        "generator.n_samples_per_prompt=5",
        "generator.gpu_memory_utilization=0.4",
        "generator.max_turns=10",
        "generator.sampling_params.temperature=0.6",
        "generator.sampling_params.top_p=0.95",
        "generator.sampling_params.stop_token_ids=[151676,151645]",
        "environment.env_class=deepanalyze",
        f"environment.skyrl_gym.deepanalyze.workspace={workspace}",
        'trainer.logger=["console","tensorboard"]',
        "trainer.project_name=deepanalyze",
        "trainer.run_name=deepanalyze_repro",
        "trainer.resume_mode=latest",
        f"trainer.ckpt_path={OUTPUT_DIR}/ckpt",
        f"trainer.export_path={OUTPUT_DIR}/export",
        "trainer.eval_batch_size=4",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=-1",
        "trainer.hf_save_interval=1",
        "trainer.ckpt_interval=1",
    ]

    print(f"\nCommand: python -m examples.deepanalyze.main_deepanalyze ...")
    print(f"Log:     {LOG_FILE}\n")

    monitor = RLMonitor()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OPENAI_API_KEY"]   = os.environ.get("OPENAI_API_KEY", "")
    env["OPENAI_BASE_URL"]  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    env["LLM_JUDGE_MODEL"]  = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

    start = time.time()
    with open(LOG_FILE, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(SKYRL_DIR),
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if any(kw in line for kw in ["Error", "Traceback", "reward", "step", "epoch"]):
                print(f"\n[RL] {line.rstrip()}")
            monitor.update(line)
        proc.wait()

    elapsed = time.time() - start
    print(f"\n\n{'='*60}")
    if proc.returncode == 0:
        print(f" Stage3 GRPO RL COMPLETE in {elapsed/60:.1f} min")
        print(f" Export: {OUTPUT_DIR}/export")
    else:
        print(f" Stage3 RL FAILED (returncode={proc.returncode}) after {elapsed/60:.1f} min")
        print(f" See log: {LOG_FILE}")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
