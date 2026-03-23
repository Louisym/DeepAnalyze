#!/bin/bash
# =============================================================
#  DeepAnalyze Full Reproduction Pipeline (Overnight)
#
#  Start:   nohup bash experiment/run_pipeline.sh > experiment/pipeline.log 2>&1 &
#  Monitor: tail -f experiment/pipeline.log
#  PID:     cat experiment/pipeline.pid
# =============================================================
set -e

EXP_DIR="/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment"
PYTHON="conda run -n base python"

# Save PID for monitoring
echo $$ > "$EXP_DIR/pipeline.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

hr() {
    echo "------------------------------------------------------"
}

log "=============================================="
log "  DeepAnalyze Reproduction Pipeline START"
log "  PID: $$"
log "=============================================="

# ── Helper: find latest ms-swift checkpoint ───────────────────────────────────
find_ckpt() {
    local base_dir="$1"
    # ms-swift saves to: base_dir/v{N}-{timestamp}/checkpoint-{step}/
    # Find the adapter_config.json anywhere under base_dir
    local adapter=$(find "$base_dir" -name "adapter_config.json" 2>/dev/null | head -1)
    if [ -n "$adapter" ]; then
        echo "$(dirname "$adapter")"
        return
    fi
    # Fallback: direct checkpoint-N subdirs
    local latest=$(ls -d "$base_dir"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "$base_dir"
    fi
}

# =============================================================
# Stage 0: Base model inference (skip if already done)
# =============================================================
hr
log "[Stage0] Base model inference"
if ls "$EXP_DIR/base_model_output"/base_result_*.json 1>/dev/null 2>&1; then
    log "[Stage0] Already exists — skipping"
else
    $PYTHON "$EXP_DIR/infer.py" --stage base
    log "[Stage0] Done"
fi

# =============================================================
# Stage 1: SFT on reasoning data
# =============================================================
hr
log "[Stage1] Starting SFT (reasoning, LoRA)..."

STAGE1_CKPT_DIR="$EXP_DIR/checkpoints/stage1_sft"
STAGE1_MERGED="$EXP_DIR/checkpoints/stage1_sft_merged"

if [ -f "$STAGE1_MERGED/config.json" ]; then
    log "[Stage1] Merged checkpoint already exists — skipping training+merge"
else
    # Skip training if adapter checkpoint already exists
    if find "$STAGE1_CKPT_DIR" -name "adapter_config.json" 2>/dev/null | grep -q .; then
        log "[Stage1] Training checkpoint found — skipping training, going straight to merge"
    else
        $PYTHON "$EXP_DIR/train_stage1_sft.py"
        log "[Stage1] Training done"
    fi

    # Merge LoRA → full model for vLLM
    log "[Stage1] Merging LoRA → $STAGE1_MERGED"
    $PYTHON "$EXP_DIR/do_merge.py" stage1
    log "[Stage1] Merge done"
fi

if ls "$EXP_DIR/sft_model_output"/sft_result_*.json 1>/dev/null 2>&1; then
    log "[Stage1] Inference output already exists — skipping"
else
    log "[Stage1] Running inference..."
    $PYTHON "$EXP_DIR/infer.py" --stage sft --model_path "$STAGE1_MERGED"
    log "[Stage1] Inference done"
fi

# =============================================================
# Stage 2: Cold-start SFT on interaction data
# =============================================================
hr
log "[Stage2] Starting cold-start SFT (interaction, LoRA)..."

STAGE2_CKPT_DIR="$EXP_DIR/checkpoints/stage2_cold"
STAGE2_MERGED="$EXP_DIR/checkpoints/stage2_cold_merged"

if [ -f "$STAGE2_MERGED/config.json" ]; then
    log "[Stage2] Merged checkpoint already exists — skipping training+merge"
else
    if find "$STAGE2_CKPT_DIR" -name "adapter_config.json" 2>/dev/null | grep -q .; then
        log "[Stage2] Training checkpoint found — skipping training, going straight to merge"
    else
        $PYTHON "$EXP_DIR/train_stage2_cold.py"
        log "[Stage2] Training done"
    fi

    # Merge LoRA → full model for vLLM
    log "[Stage2] Merging LoRA → $STAGE2_MERGED"
    $PYTHON "$EXP_DIR/do_merge.py" stage2
    log "[Stage2] Merge done"
fi

if ls "$EXP_DIR/cold_start_output"/cold_result_*.json 1>/dev/null 2>&1; then
    log "[Stage2] Inference output already exists — skipping"
else
    log "[Stage2] Running inference..."
    $PYTHON "$EXP_DIR/infer.py" --stage cold --model_path "$STAGE2_MERGED"
    log "[Stage2] Inference done"
fi

# =============================================================
# Stage 3: GRPO RL training (lightweight, no SkyRL/flash-attn)
# =============================================================
hr
log "[Stage3] Starting GRPO RL training (custom, no KL, no reference)..."
STAGE3_EXPORT="$EXP_DIR/checkpoints/stage3_rl/export"

if [ -d "$STAGE3_EXPORT" ] && [ -f "$STAGE3_EXPORT/config.json" ]; then
    log "[Stage3] Export already exists — skipping training"
else
    $PYTHON "$EXP_DIR/train_stage3_grpo_simple.py" --model_path "$STAGE2_MERGED"
    log "[Stage3] RL training done"
fi

if ls "$EXP_DIR/rl_output"/rl_result_*.json 1>/dev/null 2>&1; then
    log "[Stage3] Inference output already exists — skipping"
else
    log "[Stage3] Running inference..."
    $PYTHON "$EXP_DIR/infer.py" --stage rl --model_path "$STAGE3_EXPORT"
    log "[Stage3] Inference done"
fi

# =============================================================
# Final summary
# =============================================================
hr
log "=============================================="
log "  ALL STAGES COMPLETE"
log ""
log "  Results saved to:"
log "    Base model:   $EXP_DIR/base_model_output/"
log "    After SFT:    $EXP_DIR/sft_model_output/"
log "    After Cold:   $EXP_DIR/cold_start_output/"
log "    After RL:     $EXP_DIR/rl_output/"
log ""
log "  Compare responses by reading the *_response_*.txt files"
log "=============================================="
