"""
prepare_multilingual_dataset.py
Builds a unified, speaker-independent multilingual dataset catalog combining:
1. EasyCall (Italian): 55 speakers (31 dysarthric, 24 healthy)
2. TORGO (English): 15 speakers (8 dysarthric, 7 healthy)

Total: 70 independent human subjects across English and Italian.
Guarantees ZERO speaker overlap across train, validation, and test splits.
"""
import io
import json
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

EASYCALL_DIR = ROOT_DIR / "data" / "raw" / "easycall"
TORGO_DIR = ROOT_DIR / "data" / "raw" / "torgo"
META_DIR = ROOT_DIR / "data" / "metadata"
SAMPLE_AUDIO_DIR = ROOT_DIR / "data" / "sample_audio"

# TORGO Speaker Split Design (English)
# Total 15 speakers: 8 dysarthric, 7 healthy
TORGO_TRAIN_SPEAKERS = ["F01", "F03", "M01", "M02", "M04", "FC01", "FC02", "MC01", "MC02"] # 5 dys, 4 healthy (9 spks)
TORGO_VAL_SPEAKERS   = ["M03", "FC03"]                                                      # 1 dys, 1 healthy (2 spks)
TORGO_TEST_SPEAKERS  = ["F04", "M05", "MC03", "MC04"]                                      # 2 dys, 2 healthy (4 spks)

def build_multilingual_catalog():
    META_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # -------------------------------------------------------------------------
    # 1. EasyCall Corpus (Italian - 55 speakers)
    # -------------------------------------------------------------------------
    print("Cataloging EasyCall (Italian) corpus...")
    easycall_catalog_path = META_DIR / "dataset_catalog.csv"
    if not easycall_catalog_path.exists():
        raise FileNotFoundError("Run scripts/validate_dataset.py first for EasyCall.")
        
    df_easy = pd.read_csv(easycall_catalog_path)
    for _, row in df_easy.iterrows():
        spk = str(row['speaker']).strip().lower()
        records.append({
            "corpus": "easycall",
            "language": "it",
            "source_file": row['source_file'],
            "filename": row['filename'],
            "speaker_id": f"it_{spk}",
            "gender": "female" if spk.startswith("f") else "male",
            "class_label": row['class_label'],
            "target": int(row['target']),
            "severity": str(row['dysarthria_severity']),
            "split": row['hf_split'], # train, validation, or test
            "mic_type": "smartphone"
        })
    print(f"  Added {len(df_easy)} records from EasyCall.")
    
    # -------------------------------------------------------------------------
    # 2. TORGO Corpus (English - 15 speakers)
    # -------------------------------------------------------------------------
    print("\nCataloging TORGO (English) corpus...")
    torgo_files = sorted(list(TORGO_DIR.glob("*.parquet")))
    
    saved_en_samples = 0
    max_en_samples = 6
    
    for pfile in torgo_files:
        print(f"  Scanning {pfile.name}...")
        table = pq.read_table(pfile)
        n = len(table)
        
        audio_col = table['audio']
        statuses = table['speech_status'].to_pylist()
        genders = table['gender'].to_pylist()
        durations = table['duration'].to_pylist()
        
        for i in range(n):
            item = audio_col[i].as_py()
            path_str = item['path']
            # Parse speaker and mic from path: e.g. FC01_1_arrayMic_0066.wav
            parts = path_str.split('_')
            spk = parts[0]
            mic = parts[2] if len(parts) > 2 else "unknown"
            
            # Map status to target
            st = statuses[i].lower()
            lbl = "healthy" if st == "healthy" else "dysarthric"
            tgt = 0 if lbl == "healthy" else 1
            
            # Determine split from speaker
            if spk in TORGO_TRAIN_SPEAKERS:
                split = "train"
            elif spk in TORGO_VAL_SPEAKERS:
                split = "validation"
            elif spk in TORGO_TEST_SPEAKERS:
                split = "test"
            else:
                split = "train"
                
            # Save English sample audio files for web test presets
            if split == "test" and mic == "headMic" and saved_en_samples < max_en_samples:
                sample_name = f"torgo_en_{lbl}_{spk}_{path_str}"
                sample_path = SAMPLE_AUDIO_DIR / sample_name
                if not sample_path.exists():
                    try:
                        with io.BytesIO(item['bytes']) as bio:
                            data, sr = sf.read(bio)
                            sf.write(sample_path, data, sr)
                        saved_en_samples += 1
                    except Exception:
                        pass
                        
            records.append({
                "corpus": "torgo",
                "language": "en",
                "source_file": pfile.name,
                "filename": path_str,
                "speaker_id": f"en_{spk}",
                "gender": genders[i].lower(),
                "class_label": lbl,
                "target": tgt,
                "severity": "clinical",
                "split": split,
                "mic_type": mic
            })
            
    df_unified = pd.DataFrame(records)
    unified_catalog_path = META_DIR / "multilingual_dataset_catalog.csv"
    df_unified.to_csv(unified_catalog_path, index=False)
    
    print("\n=======================================================")
    print("MULTILINGUAL UNIFIED DATASET CATALOG SUMMARY")
    print("=======================================================")
    print(f"Total Combined Recordings: {len(df_unified)}")
    print(f"Total Unique Speakers:     {df_unified['speaker_id'].nunique()}")
    print("\nBy Language:")
    print(df_unified.groupby(['language', 'corpus', 'class_label']).agg(
        recordings=('filename', 'count'),
        speakers=('speaker_id', 'nunique')
    ))
    
    print("\nBy Split (Speaker-Independent):")
    split_summary = df_unified.groupby(['split', 'language', 'class_label']).agg(
        recordings=('filename', 'count'),
        speakers=('speaker_id', 'nunique')
    )
    print(split_summary)
    
    # Verify speaker overlap across splits
    spk_split_counts = df_unified.groupby('speaker_id')['split'].nunique()
    leaked = spk_split_counts[spk_split_counts > 1]
    if len(leaked) > 0:
        raise ValueError(f"SPEAKER LEAKAGE DETECTED! Speakers across splits: {list(leaked.index)}")
    else:
        print("\n[VERIFICATION PASSED] ZERO speaker leakage across splits in the multilingual catalog!")

    print(f"Saved multilingual catalog to: {unified_catalog_path}")

if __name__ == "__main__":
    build_multilingual_catalog()
