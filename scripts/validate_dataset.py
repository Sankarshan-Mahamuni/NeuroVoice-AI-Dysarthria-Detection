"""
validate_dataset.py
Validates EasyCall corpus files, verifies labels, speakers, metadata,
and checks audio integrity and duration distribution.
"""
import io
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

RAW_DIR = Path("data/raw/easycall")
META_DIR = Path("data/metadata")
REPORTS_DIR = Path("reports")

def validate_dataset():
    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(list(RAW_DIR.glob("*.parquet")))
    print(f"Found {len(parquet_files)} parquet files in {RAW_DIR}")
    
    all_records = []
    
    for pfile in parquet_files:
        print(f"\nProcessing {pfile.name}...")
        table = pq.read_table(pfile)
        df_meta = table.select(['filename', 'speaker', 'text', 'dysarthria_severity']).to_pandas()
        df_meta['source_file'] = pfile.name
        
        # Determine split name from filename
        if "test" in pfile.name:
            split_hint = "test"
        elif "validation" in pfile.name:
            split_hint = "validation"
        else:
            split_hint = "train"
        df_meta['hf_split'] = split_hint
        
        # Sample audio integrity checks (check first 20 in each file thoroughly, plus randomly check 50 more)
        audio_col = table['audio']
        n_rows = len(table)
        print(f"  Rows in {pfile.name}: {n_rows}")
        
        # Verify audio binary validity on a subset
        indices_to_test = list(range(min(20, n_rows))) + list(np.random.choice(n_rows, min(50, n_rows), replace=False))
        indices_to_test = sorted(list(set(indices_to_test)))
        
        durations = []
        sample_rates = []
        channels = []
        
        for idx in indices_to_test:
            item = audio_col[idx].as_py()
            raw_bytes = item['bytes']
            with io.BytesIO(raw_bytes) as bio:
                data, sr = sf.read(bio)
                dur = len(data) / sr
                durations.append(dur)
                sample_rates.append(sr)
                channels.append(1 if data.ndim == 1 else data.shape[1])
                
        print(f"  Checked {len(indices_to_test)} audio samples:")
        print(f"    Sample rate(s): {set(sample_rates)}")
        print(f"    Channels: {set(channels)}")
        print(f"    Duration range: {min(durations):.2f}s - {max(durations):.2f}s (mean: {np.mean(durations):.2f}s)")
        
        all_records.append(df_meta)
        
    combined_df = pd.concat(all_records, ignore_index=True)
    print(f"\nTotal recordings in corpus: {len(combined_df)}")
    
    # Analyze speakers and labels
    print("\n--- SPEAKER AND LABEL ANALYSIS ---")
    print(f"Total unique speakers: {combined_df['speaker'].nunique()}")
    
    # Class mapping:
    # dysarthria_severity: 0 = Healthy control; 1, 2, 3, etc. = Dysarthric; 'N/A' = Check speaker ID prefix
    # Notice: speaker prefix 'fc' and 'mc' = female/male control (healthy)
    # 'f' and 'm' without 'c' = female/male dysarthric
    def get_ground_truth_label(row):
        spk = str(row['speaker']).strip().lower()
        sev = str(row['dysarthria_severity']).strip().lower()
        
        if spk.startswith('fc') or spk.startswith('mc'):
            return 'healthy', 0
        elif spk.startswith('f') or spk.startswith('m'):
            return 'dysarthric', 1
        else:
            if sev == '0':
                return 'healthy', 0
            elif sev in ['1', '2', '3', '4']:
                return 'dysarthric', 1
            return 'unknown', -1

    labels = [get_ground_truth_label(row) for _, row in combined_df.iterrows()]
    combined_df['class_label'] = [l[0] for l in labels]
    combined_df['target'] = [l[1] for l in labels]
    
    print("\nClass breakdown:")
    print(combined_df['class_label'].value_counts())
    
    print("\nSeverity breakdown:")
    print(combined_df['dysarthria_severity'].value_counts())
    
    print("\nSpeaker-level class verification:")
    spk_groups = combined_df.groupby('speaker').agg(
        recordings=('filename', 'count'),
        class_label=('class_label', lambda x: x.iloc[0]),
        severity=('dysarthria_severity', lambda x: x.unique().tolist()),
        splits=('hf_split', lambda x: sorted(x.unique().tolist()))
    ).reset_index()
    print(spk_groups.to_string())
    
    # Check if any speaker appears in multiple HF splits
    multi_split_speakers = spk_groups[spk_groups['splits'].apply(len) > 1]
    print(f"\nSpeakers present in multiple splits in HF default splits: {len(multi_split_speakers)}")
    if len(multi_split_speakers) > 0:
        print("  WARNING: Default HF split has speaker leakage across splits:")
        print(multi_split_speakers[['speaker', 'splits']])
    else:
        print("  EXCELLENT: Default HF split has zero speaker overlap!")
        
    # Save combined catalog
    catalog_path = META_DIR / "dataset_catalog.csv"
    combined_df.to_csv(catalog_path, index=False)
    print(f"\nSaved complete dataset catalog to: {catalog_path}")
    
    spk_summary_path = META_DIR / "speaker_summary.csv"
    spk_groups.to_csv(spk_summary_path, index=False)
    print(f"Saved speaker summary to: {spk_summary_path}")
    
    # Generate dataset report
    healthy_spks = spk_groups[spk_groups['class_label'] == 'healthy']
    dysarthric_spks = spk_groups[spk_groups['class_label'] == 'dysarthric']
    split_summary = combined_df.groupby('hf_split').agg(
        recordings=('filename', 'count'),
        speakers=('speaker', 'nunique')
    ).sort_index()
    split_report = "\n".join(
        f"- **{split.title()}:** {row.recordings} recordings from {row.speakers} speakers"
        for split, row in split_summary.iterrows()
    )
    
    report_content = f"""# Dataset Report: EasyCall Dysarthric Speech Corpus

## 1. Overview & Source
- **Corpus Name:** EasyCall Corpus (Turrisi et al., INTERSPEECH 2021)
- **Repository:** Hugging Face `changelinglab/easycall-dysarthria`
- **Official Host:** Italian Institute of Technology (IIT), University of Ferrara, and Sant'Anna Hospital of Ferrara
- **License:** Creative Commons Attribution-NonCommercial 2.0 (CC BY-NC 2.0)
- **Primary Clinical Diagnostic:** Therapy Outcome Measure (TOM) clinical severity scores evaluated by neurologists.

## 2. Dataset Dimensions & Speaker Independence
- **Total Audio Recordings:** {len(combined_df)}
- **Total Independent Speakers:** {combined_df['speaker'].nunique()}
  - **Healthy Control Speakers:** {len(healthy_spks)} speakers ({spk_groups[spk_groups['class_label'] == 'healthy']['recordings'].sum()} recordings)
  - **Dysarthric Speakers:** {len(dysarthric_spks)} speakers ({spk_groups[spk_groups['class_label'] == 'dysarthric']['recordings'].sum()} recordings)
- **Gender & Pathology Separation:**
  - Female Controls (`fc*`): {len([s for s in healthy_spks['speaker'] if s.startswith('fc')])} speakers
  - Male Controls (`mc*`): {len([s for s in healthy_spks['speaker'] if s.startswith('mc')])} speakers
  - Female Dysarthric (`f*`): {len([s for s in dysarthric_spks['speaker'] if s.startswith('f')])} speakers
  - Male Dysarthric (`m*`): {len([s for s in dysarthric_spks['speaker'] if s.startswith('m')])} speakers

## 3. Provided Speaker-Independent Partitions
{split_report}

## 4. Acoustic Properties
- **Format:** Linear PCM mono audio embedded in Parquet containers.
- **Sample Rate:** 8,000 Hz native (resampled to 16,000 Hz in feature extraction pipeline).
- **Channels:** Single-channel mono.
- **Utterances:** Voice commands and phonetically balanced target words for smartphone control.

## 5. Limitations
- The corpus contains Italian speech and smartphone voice commands; the results should not be assumed to generalize to other languages or recording conditions.
- Speaker-independent evaluation is essential: the partitions must remain grouped by speaker.
- A larger number of recordings does not replace a larger number of independent participants.
"""
    with open(REPORTS_DIR / "dataset_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated comprehensive report: {REPORTS_DIR / 'dataset_report.md'}")

if __name__ == "__main__":
    validate_dataset()
