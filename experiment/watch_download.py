#!/usr/bin/env python3
"""
实时监控 HuggingFace 模型下载进度。
用法：python3 watch_download.py
"""
import os
import time
import glob

MODEL_DIR = "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B"
# 预期总大小（字节），8B 模型两个 shard 约 15.5 GB
EXPECTED_TOTAL_GB = 15.5
EXPECTED_TOTAL_BYTES = int(EXPECTED_TOTAL_GB * 1024**3)

RESET   = "\033[0m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"

def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

def bar(fraction, width=40):
    filled = int(fraction * width)
    pct = fraction * 100
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1f}%"

def get_incomplete_files():
    pattern = os.path.join(MODEL_DIR, ".cache", "huggingface", "download", "*.incomplete")
    return glob.glob(pattern)

def get_completed_safetensors():
    pattern = os.path.join(MODEL_DIR, "model*.safetensors")
    files = glob.glob(pattern)
    return {os.path.basename(f): os.path.getsize(f) for f in files}

def get_incomplete_sizes():
    files = get_incomplete_files()
    result = {}
    for f in files:
        result[os.path.basename(f)] = os.path.getsize(f)
    return result

def is_download_running():
    try:
        import subprocess
        out = subprocess.check_output(["ps", "aux"], text=True)
        return "DeepSeek" in out or "snapshot_download" in out
    except:
        return False

prev_total = 0
prev_time = time.time()
speed_history = []

print(f"\n{BOLD}{CYAN}DeepSeek-R1-0528-Qwen3-8B 下载进度监控{RESET}")
print(f"目标目录: {MODEL_DIR}")
print(f"预计总大小: {EXPECTED_TOTAL_GB} GB\n")

while True:
    os.system("clear")
    now = time.time()

    completed = get_completed_safetensors()
    incomplete = get_incomplete_sizes()
    log_tail = ""
    try:
        with open("/mnt/c/Users/louis/louis-tmp/DeepAnalyze/experiment/download_model.log") as f:
            lines = f.readlines()
            log_tail = lines[-1].strip() if lines else ""
    except:
        pass

    completed_bytes = sum(completed.values())
    incomplete_bytes = sum(incomplete.values())
    total_downloaded = completed_bytes + incomplete_bytes
    fraction = min(total_downloaded / EXPECTED_TOTAL_BYTES, 1.0)

    # 计算速度（MB/s）
    elapsed = now - prev_time
    if elapsed > 0 and prev_total > 0:
        speed_bytes = (total_downloaded - prev_total) / elapsed
        speed_history.append(speed_bytes)
        if len(speed_history) > 10:
            speed_history.pop(0)
    avg_speed = sum(speed_history) / len(speed_history) if speed_history else 0

    # 预计剩余时间
    remaining_bytes = EXPECTED_TOTAL_BYTES - total_downloaded
    if avg_speed > 0:
        eta_sec = remaining_bytes / avg_speed
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    else:
        eta_str = "计算中..."

    running = is_download_running()
    status_str = f"{GREEN}下载中{RESET}" if running else f"{YELLOW}进程未检测到（可能已完成或出错）{RESET}"

    print(f"{BOLD}{CYAN}━━━ DeepSeek-R1-0528-Qwen3-8B 下载进度 ━━━{RESET}")
    print(f"状态: {status_str}")
    print(f"时间: {time.strftime('%H:%M:%S')}\n")

    print(f"{BOLD}总体进度:{RESET}")
    print(f"  {bar(fraction)}")
    print(f"  已下载: {human_size(total_downloaded)} / {human_size(EXPECTED_TOTAL_BYTES)}")
    print(f"  速度:   {human_size(avg_speed)}/s")
    print(f"  剩余:   {eta_str}\n")

    if completed:
        print(f"{BOLD}已完成 Shard:{RESET}")
        for name, size in sorted(completed.items()):
            print(f"  {GREEN}✓{RESET} {name}  ({human_size(size)})")
        print()

    if incomplete:
        print(f"{BOLD}下载中 Shard:{RESET}")
        for name, size in sorted(incomplete.items()):
            short = name[:30] + "..." if len(name) > 33 else name
            # 每个 shard 约 7.75 GB
            shard_fraction = min(size / (7.75 * 1024**3), 1.0)
            print(f"  {YELLOW}↓{RESET} {short}")
            print(f"    {bar(shard_fraction, width=35)}  {human_size(size)}")
        print()

    if log_tail:
        print(f"{BOLD}日志最后一行:{RESET}")
        print(f"  {log_tail}")

    if fraction >= 1.0 and not incomplete:
        print(f"\n{GREEN}{BOLD}✓ 下载完成！{RESET}")
        break

    prev_total = total_downloaded
    prev_time = now

    print(f"\n  (每 5 秒刷新，Ctrl+C 退出)")
    time.sleep(5)
