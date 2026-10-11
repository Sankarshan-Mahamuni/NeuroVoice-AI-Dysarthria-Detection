"""
download_torgo.py
Downloads the English TORGO dataset from Hugging Face (abnerh/TORGO-database)
into data/raw/torgo/
"""
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "abnerh/TORGO-database"
PARQUET_FILES = [
    "data/train-00000-of-00004.parquet",
    "data/train-00001-of-00004.parquet",
    "data/train-00002-of-00004.parquet",
    "data/train-00003-of-00004.parquet",
]

RAW_DIR = Path("data/raw/torgo")

def download_torgo():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(PARQUET_FILES)} files from {REPO_ID} to {RAW_DIR}...")
    for rel_path in PARQUET_FILES:
        target_name = Path(rel_path).name
        target_file = RAW_DIR / target_name
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"  [EXISTS] {target_name} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
            continue
        print(f"  [DOWNLOADING] {rel_path}...")
        cached_file = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        shutil.copy2(cached_file, target_file)
        print(f"  [SAVED] {target_name} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
    print("\nTORGO English dataset downloaded successfully!")

if __name__ == "__main__":
    download_torgo()
