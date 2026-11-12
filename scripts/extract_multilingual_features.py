"""
extract_multilingual_features.py
Extracts features for English TORGO corpus and merges with Italian EasyCall features
to produce unified multilingual feature matrices (English + Italian).
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.audio_preprocessing import load_and_validate_audio
from app.services.feature_extraction import extract_acoustic_features, FEATURE_DIM

TORGO_DIR = ROOT_DIR / "data" / "raw" / "torgo"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
META_DIR = ROOT_DIR / "data" / "metadata"

TORGO_TRAIN_SPEAKERS = ["F01", "F03", "M01", "M02", "M04", "FC01", "FC02", "MC01", "MC02"]
TORGO_VAL_SPEAKERS   = ["M03", "FC03"]
TORGO_TEST_SPEAKERS  = ["F04", "M05", "MC03", "MC04"]

def process_torgo_single(item_tuple):
    idx, fn, spk, lbl, tgt, split, raw_bytes = item_tuple
    try:
        signal, sr, meta = load_and_validate_audio(raw_bytes)
        features = extract_acoustic_features(signal, sr)
        return (idx, fn, spk, lbl, tgt, split, features, meta, None)
    except Exception as e:
        return (idx, fn, spk, lbl, tgt, split, None, None, str(e))

def extract_and_merge():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract TORGO features
    torgo_files = sorted(list(TORGO_DIR.glob("*.parquet")))
    print(f"Reading TORGO files ({len(torgo_files)} partitions)...")
    
    torgo_items = {"train": [], "validation": [], "test": []}
    
    for pfile in torgo_files:
        table = pq.read_table(pfile)
        n = len(table)
        audio_col = table['audio']
        statuses = table['speech_status'].to_pylist()
        
        for i in range(n):
            item = audio_col[i].as_py()
            path_str = item['path']
            spk = path_str.split('_')[0]
            st = statuses[i].lower()
            lbl = "healthy" if st == "healthy" else "dysarthric"
            tgt = 0 if lbl == "healthy" else 1
            
            if spk in TORGO_TRAIN_SPEAKERS:
                split = "train"
            elif spk in TORGO_VAL_SPEAKERS:
                split = "validation"
            elif spk in TORGO_TEST_SPEAKERS:
                split = "test"
            else:
                split = "train"
                
            torgo_items[split].append((i, path_str, spk, lbl, tgt, split, item['bytes']))
            
    torgo_extracted = {}
    for split_name in ["train", "validation", "test"]:
        items = torgo_items[split_name]
        print(f"\nExtracting TORGO (English) {split_name.upper()} ({len(items)} samples)...")
        start_t = time.time()
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(process_torgo_single, items, chunksize=64), total=len(items), desc=f"TORGO {split_name}"))
            
        X_list, y_list, spk_list, lang_list = [], [], [], []
        dropped = 0
        for res in results:
            idx, fn, spk, lbl, tgt, split, feat, meta, err = res
            if err is not None:
                dropped += 1
                continue
            X_list.append(feat)
            y_list.append(tgt)
            spk_list.append(f"en_{spk}")
            lang_list.append("en")
            
        elapsed = time.time() - start_t
        print(f"  Extracted {len(X_list)} English samples in {elapsed:.1f}s ({dropped} dropped)")
        torgo_extracted[split_name] = {
            "X": np.array(X_list, dtype=np.float32),
            "y": np.array(y_list, dtype=np.int64),
            "spk": np.array(spk_list, dtype=object),
            "lang": np.array(lang_list, dtype=object)
        }
        
    # 2. Merge with existing Italian EasyCall features
    print("\nMerging with Italian EasyCall features to create unified multilingual datasets...")
    for split_name in ["train", "validation", "test"]:
        it_X = np.load(PROCESSED_DIR / f"{split_name}_features.npy")
        it_y = np.load(PROCESSED_DIR / f"{split_name}_labels.npy")
        it_spk = np.load(PROCESSED_DIR / f"{split_name}_speakers.npy", allow_pickle=True)
        it_spk = np.array([f"it_{s}" for s in it_spk], dtype=object)
        it_lang = np.array(["it"] * len(it_X), dtype=object)
        
        en_data = torgo_extracted[split_name]
        
        # Stack English + Italian
        combined_X = np.vstack([it_X, en_data["X"]])
        combined_y = np.concatenate([it_y, en_data["y"]])
        combined_spk = np.concatenate([it_spk, en_data["spk"]])
        combined_lang = np.concatenate([it_lang, en_data["lang"]])
        
        # Shuffle synchronously to interleave languages
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(combined_X))
        combined_X = combined_X[perm]
        combined_y = combined_y[perm]
        combined_spk = combined_spk[perm]
        combined_lang = combined_lang[perm]
        
        # Save multilingual files
        np.save(PROCESSED_DIR / f"multi_{split_name}_features.npy", combined_X)
        np.save(PROCESSED_DIR / f"multi_{split_name}_labels.npy", combined_y)
        np.save(PROCESSED_DIR / f"multi_{split_name}_speakers.npy", combined_spk)
        np.save(PROCESSED_DIR / f"multi_{split_name}_langs.npy", combined_lang)
        
        print(f"\n[SAVED] multi_{split_name}_features.npy: {combined_X.shape}")
        print(f"  Class balance: Healthy={np.sum(combined_y == 0)}, Dysarthric={np.sum(combined_y == 1)}")
        print(f"  Language balance: English={np.sum(combined_lang == 'en')}, Italian={np.sum(combined_lang == 'it')}")
        print(f"  Unique speakers in {split_name}: {len(np.unique(combined_spk))}")

    print("\nMultilingual feature extraction and merging complete!")

if __name__ == "__main__":
    extract_and_merge()
