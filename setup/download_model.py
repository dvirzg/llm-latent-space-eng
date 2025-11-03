#!/usr/bin/env python3
"""
Download gpt-oss-20b from HuggingFace Hub (~40GB)
"""

from huggingface_hub import snapshot_download
import os

MODEL_NAME = "openai/gpt-oss-20b"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading {MODEL_NAME} w/ cache to {CACHE_DIR}")

try:
    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        resume_download=True,  # Resume if interrupted
        local_files_only=False,
    )
    print(f"\nModel downloaded successfully.")
    print(f"Location: {model_path}")
except Exception as e:
    print(f"\nError downloading model: {e}")
