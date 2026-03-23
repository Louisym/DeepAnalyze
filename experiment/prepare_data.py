#!/usr/bin/env python3
"""
Sample training data for each stage and convert to ms-swift format.

Stage1 (SFT reasoning):    2000 samples from reasoning/ JSON files
Stage2 (cold-start SFT):   1200 samples from interation/ JSON files
Stage3 (GRPO RL):           600 samples from RL/ parquet files
"""
import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Load .env
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))
EXP_DIR  = Path(os.environ.get("EXPERIMENT_DIR", ""))

SEED = 42
random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} items → {path}")

def sample_from_files(file_list, total_n):
    """Load all files, then randomly sample total_n items."""
    all_items = []
    for fpath in file_list:
        items = load_json(fpath)
        all_items.extend(items)
        print(f"  Loaded {len(items):>5} items from {Path(fpath).name}")
    print(f"  Total pool: {len(all_items)} items, sampling {total_n}")
    return random.sample(all_items, min(total_n, len(all_items)))


# ── Stage 1: Reasoning SFT data ───────────────────────────────────────────────

def prepare_stage1():
    print("\n" + "="*60)
    print("Stage1: Reasoning SFT data (target: 2000 samples)")
    print("="*60)

    reasoning_dir = DATA_DIR / "reasoning"
    files = sorted(reasoning_dir.glob("*.json"))
    print(f"Found {len(files)} reasoning files")

    samples = sample_from_files(files, 2000)

    # Convert to ms-swift format: {"messages": [...]}
    swift_data = []
    for item in tqdm(samples, desc="  Converting"):
        swift_data.append({"messages": item["messages"]})

    out = EXP_DIR / "data" / "stage1" / "train.jsonl"
    save_jsonl(swift_data, out)
    return len(swift_data)


# ── Stage 2: Cold-start interaction SFT data ──────────────────────────────────

def prepare_stage2():
    print("\n" + "="*60)
    print("Stage2: Cold-start SFT data (target: 1200 samples)")
    print("="*60)

    inter_dir = DATA_DIR / "interation"
    files = sorted(inter_dir.glob("*.json"))
    print(f"Found {len(files)} interaction files")

    samples = sample_from_files(files, 1200)

    swift_data = []
    for item in tqdm(samples, desc="  Converting"):
        swift_data.append({"messages": item["messages"]})

    out = EXP_DIR / "data" / "stage2" / "train.jsonl"
    save_jsonl(swift_data, out)
    return len(swift_data)


# ── Stage 3: GRPO RL data ─────────────────────────────────────────────────────

def prepare_stage3():
    print("\n" + "="*60)
    print("Stage3: GRPO RL data (target: 600 samples)")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas not available. Run: pip install pandas pyarrow")
        return 0

    rl_dir = DATA_DIR / "RL"
    parquet_files = sorted(rl_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    all_rows = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        print(f"  Loaded {len(df):>5} rows from {pf.name}, cols: {list(df.columns)}")
        all_rows.append(df)

    import pandas as pd
    df_all = pd.concat(all_rows, ignore_index=True)
    print(f"  Total pool: {len(df_all)} rows, sampling 600")

    df_sample = df_all.sample(n=min(600, len(df_all)), random_state=SEED)

    out_dir = EXP_DIR / "data" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "train.parquet"
    df_sample.to_parquet(out_parquet, index=False)
    print(f"  Saved {len(df_sample)} rows → {out_parquet}")

    # Also save as jsonl for inspection
    out_jsonl = out_dir / "train_preview.jsonl"
    records = df_sample.to_dict(orient="records")
    with open(out_jsonl, "w") as f:
        for r in records[:20]:  # only first 20 for preview
            f.write(json.dumps({k: str(v)[:200] for k,v in r.items()}, ensure_ascii=False) + "\n")
    print(f"  Preview (first 20) → {out_jsonl}")
    return len(df_sample)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"EXP_DIR:  {EXP_DIR}")

    n1 = prepare_stage1()
    n2 = prepare_stage2()
    n3 = prepare_stage3()

    print("\n" + "="*60)
    print("Data preparation complete!")
    print(f"  Stage1: {n1} samples → experiment/data/stage1/train.jsonl")
    print(f"  Stage2: {n2} samples → experiment/data/stage2/train.jsonl")
    print(f"  Stage3: {n3} samples → experiment/data/stage3/train.parquet")
    print("="*60)


if __name__ == "__main__":
    main()
