# DeepAnalyze Reproduction — A800 Cluster

## Project Overview

Reproducing the DeepAnalyze paper's 4-stage training pipeline:
- **Paper**: "DeepAnalyze: Agentic Large Language Models for Autonomous Data Science"
- **Base model**: DeepSeek-R1-0528-Qwen3-8B
- **Training data**: DataScience-Instruct-500K (reasoning / interaction / RL subsets)
- **Evaluation**: DABStep-Research benchmark with LLM judge + checklist scoring

## 4-Stage Pipeline

| Stage | Type | Data | Description |
|-------|------|------|-------------|
| Stage0 | Base inference | - | Baseline: run base model on test task, save response |
| Stage1 | Full SFT | `reasoning/` (13,023 samples) | Supervised fine-tune on reasoning data |
| Stage2 | Cold-start SFT | `interation/` (~10,941 samples) | SFT on interaction/code-execution data (builds on Stage1 output) |
| Stage3 | GRPO RL | `RL/` parquets | Group Relative Policy Optimization (builds on Stage2 output) |

## Current Task: Stage1 Full SFT on A800

### What to do

Run full SFT (NOT LoRA) on ALL reasoning data using a single A800 80GB GPU.

### Key decisions already made

- **Full fine-tune** (not LoRA) — A800 80GB has enough memory
- **All 13,023 reasoning samples** — use everything, no sampling
- **max_length=8192** — critical! Old RTX 5090 run used 1792 which truncated 63.6% of data. Token length stats:
  - P50=2283, P95=4429, P99=6436, max=144692
  - 8192 covers 99.3% of data; filter out the 85 extreme outlier samples (>8192 tokens)
- **Save only final checkpoint** — no intermediate checkpoints; this model will be the base for Stage2 cold-start and Stage3 RL
- **After training completes**: run inference on the student loan test task and save the response for quality inspection

### Training hyperparameters

```
train_type:     full (NOT lora)
max_length:     8192
batch_size:     2 (per device)
grad_accum:     8 (effective batch = 16)
learning_rate:  1e-5 (lower than LoRA's 5e-5, appropriate for full SFT)
epochs:         1
warmup_ratio:   0.05
total_steps:    ~808 (12938 samples / 16 effective batch)
attn_impl:      flash_attn
packing:        false
gradient_checkpointing: true
```

### Reference script

See `experiment/train_stage1_sft_a800.py` — written for A800 cluster. Key things to adapt:
- Paths at the top (BASE_MODEL, DATA_DIR, EXP_DIR, WORKSPACE_DIR) — match your cluster layout
- GPU_ID — set to whichever GPU is free
- The script has 3 steps: (1) prepare all reasoning data → JSONL, (2) run ms-swift `swift sft`, (3) run vLLM inference on student loan task

### How to run

```bash
# Option A: Run the existing script (edit paths first)
CUDA_VISIBLE_DEVICES=7 nohup python experiment/train_stage1_sft_a800.py > experiment/sft_a800.log 2>&1 &

# Option B: Rewrite from scratch using ms-swift CLI directly
swift sft \
  --model <base_model_path> \
  --train_type full \
  --dataset <train.jsonl> \
  --max_length 8192 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --gradient_checkpointing true \
  --attn_impl flash_attn \
  --model_type qwen3 \
  --torch_dtype bfloat16 \
  --truncation_strategy right \
  --output_dir <output_dir>
```

## Data Format

Training data is ms-swift format JSONL:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Source files in `DataScience-Instruct-500K/reasoning/`:
- `file_csv_3007.json` (3,007 samples)
- `file_database_3833.json` (3,833 samples)
- `file_xlsx_3663.json` (3,663 samples)
- `file_any_2520.json` (2,520 samples)

Each sample has `messages`, `input_tokens`, `output_tokens`, `total_tokens`, `evaluation` fields. Only `messages` is needed for training.

## Special Tokens (added via add_vocab)

The base model needs 10 special tokens added to tokenizer:
`<Analyze>`, `</Analyze>`, `<Code>`, `</Code>`, `<Execute>`, `</Execute>`, `<Understand>`, `</Understand>`, `<Answer>`, `</Answer>` (IDs 151671–151680)

Script: `experiment/add_vocab_light.py` — modifies tokenizer JSON only (no model weight loading), hardlinks safetensors files.

If the base model on the cluster already has these tokens in its tokenizer, skip this step. If not, run `add_vocab_light.py` first to create a `*-addvocab` version.

## Inference (after training)

Use vLLM for inference on the student loan test task:
```python
from vllm import LLM, SamplingParams
llm = LLM(model=checkpoint_path, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=7000, stop=["</Answer>", "<|im_end|>"])
```

Test task data is in `example/analysis_on_student_loan/data/` (multiple CSV files about student loans).

## DABStep-Research Evaluation

Located in `playground/DABStep-Research/`:
- `dabstep_research.jsonl` — benchmark tasks with checklists
- `run_deepanalyze.py` — runs model on tasks
- `gpt_eval.py` — LLM judge scoring (Content 1-5 + Format 1-5, averaged)
- Checklist is NOT scored per-item; it's passed as context to the LLM judge as "reference points for an ideal report"

## Dependencies

- ms-swift (`pip install ms-swift`)
- flash-attn (should work on A800 + CUDA 12.8)
- vllm
- transformers, peft, bitsandbytes
- openai (for LLM judge in RL stage)

## Lessons Learned from RTX 5090 Run

1. **max_length=1792 was disastrous** — truncated 63.6% of data, resulting in only 11 effective training steps. Use 8192.
2. **ms-swift quirks**: requires `--model_type qwen3`, crashes without `QT_QPA_PLATFORM=offscreen`, saves checkpoints to `output_dir/v{N}-{timestamp}/checkpoint-{step}/` (nested, not flat)
3. **vLLM inference needs `stop=["</Answer>", "<|im_end|>"]`** — without `<|im_end|>` the model may not stop properly
4. **LoRA merge**: use PEFT API `model.merge_and_unload()` directly, NOT `swift export` (which fails if output dir exists)
5. **Full SFT uses lower LR** (1e-5) than LoRA (5e-5) to avoid catastrophic forgetting

## After Stage1

Once Stage1 SFT is done and verified:
- Stage2 cold-start SFT: train on `interation/` data using Stage1 checkpoint as base
- Stage3 GRPO RL: train on `RL/` data using Stage2 checkpoint as base
- Each stage's output is the next stage's input model
