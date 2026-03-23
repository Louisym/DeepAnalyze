#!/usr/bin/env python3
"""
Add special tokens to tokenizer ONLY (no model weights loaded).
Copies all model files and updates tokenizer_config + tokenizer.json.
Much faster and uses minimal RAM.
"""
import os
import json
import shutil
from pathlib import Path

MODEL_IN  = "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B"
MODEL_OUT = "/mnt/c/Users/louis/louis-tmp/models/DeepSeek-R1-0528-Qwen3-8B-addvocab"

NEW_TOKENS = [
    "<Analyze>", "</Analyze>",
    "<Understand>", "</Understand>",
    "<Code>", "</Code>",
    "<Execute>", "</Execute>",
    "<Answer>", "</Answer>",
]

def main():
    src = Path(MODEL_IN)
    dst = Path(MODEL_OUT)

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Copy model weight files (hardlink to save disk space)
    print("\n[1/4] Linking model weight files...")
    for f in src.iterdir():
        if f.is_dir():
            print(f"  skip (dir): {f.name}")
            continue
        dst_f = dst / f.name
        if dst_f.exists():
            print(f"  skip (exists): {f.name}")
            continue
        if f.suffix in (".safetensors", ".bin") or f.name == "model.safetensors.index.json":
            try:
                os.link(f, dst_f)
                print(f"  hardlink: {f.name}")
            except OSError:
                shutil.copy2(f, dst_f)
                print(f"  copy: {f.name}")
        else:
            shutil.copy2(f, dst_f)
            print(f"  copy: {f.name}")

    # 2. Load tokenizer files
    print("\n[2/4] Loading tokenizer files...")
    tok_config_path = dst / "tokenizer_config.json"
    tok_path = dst / "tokenizer.json"

    with open(tok_config_path) as f:
        tok_config = json.load(f)
    with open(tok_path) as f:
        tok_json = json.load(f)

    # 3. Add new tokens to tokenizer.json
    print("\n[3/4] Adding special tokens...")
    added_tokens = tok_json.get("added_tokens", [])
    existing = {t["content"] for t in added_tokens}

    # Find current max id
    vocab = tok_json.get("model", {}).get("vocab", {})
    all_ids = list(vocab.values()) + [t["id"] for t in added_tokens]
    next_id = max(all_ids) + 1 if all_ids else 151936

    added_count = 0
    for token in NEW_TOKENS:
        if token in existing:
            print(f"  already exists: {token}")
            continue
        added_tokens.append({
            "id": next_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })
        print(f"  added id={next_id}: {token}")
        next_id += 1
        added_count += 1

    tok_json["added_tokens"] = added_tokens

    # Also add to post_processor / special_tokens if needed
    # Update tokenizer_config additional_special_tokens
    existing_special = tok_config.get("additional_special_tokens", [])
    for token in NEW_TOKENS:
        if token not in existing_special:
            existing_special.append(token)
    tok_config["additional_special_tokens"] = existing_special

    # 4. Save updated tokenizer files
    print("\n[4/4] Saving updated tokenizer...")
    with open(tok_path, "w") as f:
        json.dump(tok_json, f, ensure_ascii=False, indent=2)
    with open(tok_config_path, "w") as f:
        json.dump(tok_config, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Added {added_count} new tokens.")
    print(f"New vocab size: {next_id}")
    print(f"Saved to: {dst}")

    # Verify
    print("\nVerifying with transformers...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(dst))
    for token in NEW_TOKENS:
        tid = tok.convert_tokens_to_ids(token)
        print(f"  {token!r:20s} → id={tid}")
    print(f"\nVocab size (tokenizer): {len(tok)}")

if __name__ == "__main__":
    main()
