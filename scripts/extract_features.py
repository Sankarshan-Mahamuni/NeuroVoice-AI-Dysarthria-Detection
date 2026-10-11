"""
extract_features.py
Extracts acoustic features across the speaker-independent splits (train, validation, test)
using the shared preprocessing & feature extraction modules.
Utilizes multi-core processing to process recordings rapidly.
Also extracts sample audio files to data/sample_audio/ for web interface demonstration.
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from app.services.audio_preprocessing import load_and_validate_audio, AudioProcessingError
from app.services.feature_extraction import extract_acoustic_features, FEATURE_DIM

RAW_DIR = Path("data/raw/easycall")
PROCESSED_DIR = Path("data/processed")
META_DIR = Path("data/metadata")
SAMPLE_AUDIO_DIR = Path("data/sample_audio")

def process_single_item(item_tuple):
    """Worker function to process a single audio byte stream."""
    idx, filename, speaker, label_name, target, severity, split, raw_bytes = item_tuple
    try:
        signal, sr, meta = load_and_validate_audio(raw_bytes)
        features = extract_acoustic_features(signal, sr)
        return (idx, filename, speaker, label_name, target, severity, split, features, meta, None)
    except Exception as e:
        return (idx, filename, speaker, label_name, target, severity, split, None, None, str(e))

def extract_all_features():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    catalog_path = META_DIR / "dataset_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError("Run scripts/validate_dataset.py first to generate dataset_catalog.csv")
        
    catalog_df = pd.read_csv(catalog_path)
    print(f"Loaded catalog with {len(catalog_df)} records.")
    
    # Process each split separately
    splits = ["train", "validation", "test"]
    split_file_map = {
        "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
        "validation": ["validation-00000-of-00001.parquet"],
        "test": ["test-00000-of-00001.parquet"]
    }
    
    saved_samples_count = 0
    sampled_speakers = {0: set(), 1: set()}
    max_samples_per_class = 5
    extraction_failures = []
    
    for split_name in splits:
        print(f"\n==========================================")
        print(f"Processing split: {split_name.upper()}")
        print(f"==========================================")
        
        target_files = split_file_map[split_name]
        items_to_process = []
        
        for pfilename in target_files:
            pfile = RAW_DIR / pfilename
            print(f"Loading table from {pfile.name}...")
            table = pq.read_table(pfile)
            n_rows = len(table)
            
            audio_col = table['audio']
            fn_col = table['filename']
            spk_col = table['speaker']
            sev_col = table['dysarthria_severity']
            
            for i in range(n_rows):
                fn = str(fn_col[i].as_py())
                spk = str(spk_col[i].as_py())
                sev = str(sev_col[i].as_py())
                raw_bytes = audio_col[i].as_py()['bytes']
                
                # Determine label
                if spk.startswith('fc') or spk.startswith('mc'):
                    lbl, tgt = 'healthy', 0
                else:
                    lbl, tgt = 'dysarthric', 1
                    
                # Save at most one example per validation speaker and class for the demo UI.
                if (
                    split_name == "validation"
                    and len(sampled_speakers[tgt]) < max_samples_per_class
                    and spk not in sampled_speakers[tgt]
                ):
                    sample_wav_path = SAMPLE_AUDIO_DIR / f"{lbl}_{spk}_{fn}"
                    try:
                        if not sample_wav_path.exists():
                            with io.BytesIO(raw_bytes) as bio:
                                data, sr = sf.read(bio)
                            sf.write(sample_wav_path, data, sr)
                        sampled_speakers[tgt].add(spk)
                        saved_samples_count += 1
                    except (OSError, RuntimeError, ValueError) as sample_error:
                        print(f"  Could not save demo sample {sample_wav_path.name}: {sample_error}")
                            
                items_to_process.append((i, fn, spk, lbl, tgt, sev, split_name, raw_bytes))
                
        print(f"Total recordings to extract for {split_name}: {len(items_to_process)}")
        start_time = time.time()
        
        # Parallel extraction using ProcessPoolExecutor
        X_list = []
        y_list = []
        speakers_list = []
        meta_records = []
        dropped_count = 0
        
        # Run with max_workers=8 to balance memory and CPU throughput
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(process_single_item, items_to_process, chunksize=64), total=len(items_to_process), desc=f"Extracting {split_name}"))
            
        for res in results:
            idx, fn, spk, lbl, tgt, sev, split, feat, meta, err = res
            if err is not None:
                dropped_count += 1
                extraction_failures.append({
                    "split": split,
                    "speaker": spk,
                    "filename": fn,
                    "error": err
                })
                continue
            X_list.append(feat)
            y_list.append(tgt)
            speakers_list.append(spk)
            meta_dict = {
                "filename": fn,
                "speaker": spk,
                "label": lbl,
                "target": tgt,
                "severity": sev,
                "split": split,
                "duration_sec": meta["duration_sec"],
                "peak_amp": meta["peak_amplitude"],
                "rms_energy": meta["rms_energy"]
            }
            meta_records.append(meta_dict)
            
        elapsed = time.time() - start_time
        print(f"Extraction completed for {split_name} in {elapsed:.1f}s ({len(X_list)/elapsed:.1f} samples/sec)")
        if dropped_count > 0:
            print(f"  Dropped {dropped_count} corrupted or invalid samples.")
            
        # Convert to numpy arrays
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int64)
        spk_arr = np.array(speakers_list, dtype=object)
        meta_df = pd.DataFrame(meta_records)
        
        # Save artifacts
        np.save(PROCESSED_DIR / f"{split_name}_features.npy", X_arr)
        np.save(PROCESSED_DIR / f"{split_name}_labels.npy", y_arr)
        np.save(PROCESSED_DIR / f"{split_name}_speakers.npy", spk_arr)
        meta_df.to_csv(PROCESSED_DIR / f"{split_name}_metadata.csv", index=False)
        
        print(f"  Saved {split_name}_features.npy: shape {X_arr.shape}")
        print(f"  Class balance: Healthy={np.sum(y_arr == 0)}, Dysarthric={np.sum(y_arr == 1)}")
        print(f"  Unique speakers in {split_name}: {len(np.unique(spk_arr))}")

    # Generate feature names list for interpretability
    feature_names = []
    # MFCC mean & std (20)
    for i in range(20): feature_names.append(f"mfcc_{i+1}_mean")
    for i in range(20): feature_names.append(f"mfcc_{i+1}_std")
    # Delta mean & std (20)
    for i in range(20): feature_names.append(f"delta_mfcc_{i+1}_mean")
    for i in range(20): feature_names.append(f"delta_mfcc_{i+1}_std")
    # Delta2 mean & std (20)
    for i in range(20): feature_names.append(f"delta2_mfcc_{i+1}_mean")
    for i in range(20): feature_names.append(f"delta2_mfcc_{i+1}_std")
    # Spectral
    feature_names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
    feature_names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])
    feature_names.extend(["spectral_rolloff_mean", "spectral_rolloff_std"])
    feature_names.extend(["spectral_flatness_mean", "spectral_flatness_std"])
    for i in range(6): feature_names.append(f"spectral_contrast_band{i+1}_mean")
    for i in range(6): feature_names.append(f"spectral_contrast_band{i+1}_std")
    # Prosodic / Energy
    feature_names.extend(["rms_energy_mean", "rms_energy_std"])
    feature_names.extend(["zero_crossing_rate_mean", "zero_crossing_rate_std"])
    
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    failure_log_path = META_DIR / "feature_extraction_failures.csv"
    pd.DataFrame(
        extraction_failures,
        columns=["split", "speaker", "filename", "error"]
    ).to_csv(failure_log_path, index=False)
    print(f"\nSaved {len(feature_names)} feature names to {PROCESSED_DIR / 'feature_names.json'}")
    print(f"Saved {saved_samples_count} sample audio files to {SAMPLE_AUDIO_DIR}")
    print(f"Saved {len(extraction_failures)} extraction failures to {failure_log_path}")

if __name__ == "__main__":
    extract_all_features()
