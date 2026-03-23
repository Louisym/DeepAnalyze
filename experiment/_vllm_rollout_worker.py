#!/usr/bin/env python3
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
