#!/bin/bash
# Setup conda environments for DeepAnalyze reproduction
# Run: bash experiment/setup_envs.sh [da-infer|da-sft|all]
# Each section is idempotent - safe to re-run

set -e

STEP=${1:-all}

echo "============================================"
echo " DeepAnalyze Environment Setup"
echo " Step: $STEP"
echo "============================================"

# ─── da-infer: vLLM for inference ────────────────────────────────────────────
setup_infer() {
    echo ""
    echo "[1/2] Creating da-infer (vLLM inference env)..."
    if conda env list | grep -q "^da-infer "; then
        echo "  da-infer already exists, skipping create"
    else
        conda create -n da-infer python=3.11 -y
    fi

    echo "  Installing vLLM..."
    conda run -n da-infer pip install vllm==0.9.0.1 --index-url https://pypi.org/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        2>&1 | tee /tmp/da-infer-install.log | grep -E "(Collecting|Installing|Successfully|ERROR|error)" || true

    echo "  Installing additional packages..."
    conda run -n da-infer pip install transformers accelerate tqdm python-dotenv openai 2>&1 | \
        grep -E "(Collecting|Installing|Successfully|ERROR)" || true

    echo "  [da-infer] Verifying..."
    conda run -n da-infer python -c "import vllm; print('  vLLM version:', vllm.__version__)"
    echo "  [da-infer] OK"
}

# ─── da-sft: ms-swift for SFT training ───────────────────────────────────────
setup_sft() {
    echo ""
    echo "[2/2] Creating da-sft (ms-swift SFT env)..."
    if conda env list | grep -q "^da-sft "; then
        echo "  da-sft already exists, skipping create"
    else
        conda create -n da-sft python=3.11 -y
    fi

    SWIFT_DIR="/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/ms-swift"

    echo "  Installing PyTorch with CUDA 12.8..."
    conda run -n da-sft pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        2>&1 | grep -E "(Collecting|Installing|Successfully|ERROR)" || true

    echo "  Installing ms-swift from local source..."
    conda run -n da-sft pip install -e "$SWIFT_DIR" 2>&1 | \
        grep -E "(Collecting|Installing|Successfully|ERROR)" || true

    echo "  Installing additional packages..."
    conda run -n da-sft pip install deepspeed tqdm python-dotenv openai 2>&1 | \
        grep -E "(Collecting|Installing|Successfully|ERROR)" || true

    echo "  [da-sft] Verifying..."
    conda run -n da-sft python -c "import swift; print('  ms-swift version:', swift.__version__)"
    echo "  [da-sft] OK"
}

case $STEP in
    da-infer) setup_infer ;;
    da-sft)   setup_sft ;;
    all)      setup_infer; setup_sft ;;
    *)        echo "Unknown step: $STEP. Use: da-infer | da-sft | all"; exit 1 ;;
esac

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
