"""
download_dataset.py
Script to reproducibly download the EasyCall dysarthria corpus from Hugging Face Hub.
"""
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pandas as pd

REPO_ID = "changelinglab/easycall-dysarthria"
PARQUET_FILES = [
    "data/test-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]

RAW_DIR = Path("data/raw/easycall")

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []
    
    print(f"Downloading {len(PARQUET_FILES)} parquet partitions from {REPO_ID}...")
    for rel_path in PARQUET_FILES:
        target_name = Path(rel_path).name
        target_file = RAW_DIR / target_name
        
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f" [EXISTS] {target_name} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
            downloaded_paths.append(target_file)
            continue
            
        print(f" [DOWNLOADING] {rel_path} -> {target_file}...")
        cached_file = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        shutil.copy2(cached_file, target_file)
        print(f" [SAVED] {target_name} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        downloaded_paths.append(target_file)
        
    print("\nAll raw dataset files successfully downloaded to:", RAW_DIR)
    return downloaded_paths

if __name__ == "__main__":
    download_dataset()
