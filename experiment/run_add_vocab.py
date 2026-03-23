#!/usr/bin/env python3
"""Run add_vocab.py with progress monitoring."""
import subprocess
import sys
import time
from datetime import datetime

MODEL_IN = "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B"
MODEL_OUT = "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B-addvocab"
SCRIPT = "/mnt/c/Users/louis/louis-tmp/DeepAnalyze/deepanalyze/add_vocab.py"

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting add_vocab...")
print(f"  Input:  {MODEL_IN}")
print(f"  Output: {MODEL_OUT}")
print()

start = time.time()
proc = subprocess.Popen(
    [sys.executable, SCRIPT,
     "--model_path", MODEL_IN,
     "--save_path", MODEL_OUT,
     "--add_tags"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in proc.stdout:
    elapsed = time.time() - start
    print(f"[{elapsed:6.1f}s] {line}", end="")

proc.wait()
elapsed = time.time() - start
if proc.returncode == 0:
    print(f"\n[{elapsed:.1f}s] add_vocab DONE - model saved to {MODEL_OUT}")
else:
    print(f"\n[{elapsed:.1f}s] add_vocab FAILED (returncode={proc.returncode})")
    sys.exit(1)
