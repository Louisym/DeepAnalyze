#!/bin/bash
# Install SkyRL dependencies into base conda env
# Run once before Stage3 RL training
set -e

SKYRL_DIR="/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/SkyRL"
SKYRL_TRAIN="$SKYRL_DIR/skyrl-train"
SKYRL_GYM="$SKYRL_DIR/skyrl-gym"

echo "[install_skyrl] Installing skyrl-gym..."
conda run -n base pip install -e "$SKYRL_GYM" 2>&1 | grep -E "(Collecting|Installing|Successfully|ERROR)" || true

echo "[install_skyrl] Installing skyrl-train..."
conda run -n base pip install -e "$SKYRL_TRAIN" --no-deps 2>&1 | grep -E "(Collecting|Installing|Successfully|ERROR)" || true

echo "[install_skyrl] Installing hydra-core and other deps..."
conda run -n base pip install "hydra-core==1.3.2" omegaconf torchdata tensordict jaxtyping func_timeout loguru 2>&1 | grep -E "(Collecting|Installing|Successfully|ERROR)" || true

# Downgrade ray to match SkyRL requirement (2.48.0)
echo "[install_skyrl] Fixing ray version (need 2.48.0)..."
conda run -n base pip install "ray==2.48.0" 2>&1 | grep -E "(Collecting|Installing|Successfully|ERROR|Uninstalling)" || true

echo "[install_skyrl] Verifying..."
conda run -n base python -c "
import skyrl_gym; print('skyrl-gym: OK')
import skyrl_train; print('skyrl-train: OK')
import hydra; print('hydra:', hydra.__version__)
import ray; print('ray:', ray.__version__)
" 2>&1

echo "[install_skyrl] Done"
