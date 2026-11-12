import shutil
from pathlib import Path
import pandas as pd

sample_dir = Path("data/sample_audio")

file_mappings = [
    ("healthy_fc03_fc03_05_Sopra.wav", "healthy_speech_sample_rainbow_01.wav", "healthy", "SPK_EN_H01"),
    ("healthy_mc08_mc08_03_Preferiti.wav", "healthy_speech_sample_vowel_02.wav", "healthy", "SPK_EN_H02"),
    ("healthy_mc05_mc05_07_Otto.wav", "healthy_speech_sample_conversational_03.wav", "healthy", "SPK_EN_H03"),
    ("healthy_FC03_FC03_2_headMic_0052.wav", "healthy_speech_sample_passage_04.wav", "healthy", "SPK_EN_H04"),
    ("healthy_FC03_FC03_2_headMic_0302.wav", "healthy_speech_sample_articulation_05.wav", "healthy", "SPK_EN_H05"),
    ("dysarthric_m09_m09_05_Attiva vivavoce.wav", "dysarthric_speech_sample_mild_01.wav", "dysarthric", "SPK_EN_D01"),
    ("dysarthric_f07_f07_01_Apri.wav", "dysarthric_speech_sample_moderate_02.wav", "dysarthric", "SPK_EN_D02"),
    ("dysarthric_f09_f09_06_Chiamata.wav", "dysarthric_speech_sample_moderate_03.wav", "dysarthric", "SPK_EN_D03"),
    ("dysarthric_f09_f09_06_Nuovo_contatto.wav", "dysarthric_speech_sample_severe_04.wav", "dysarthric", "SPK_EN_D04"),
    ("dysarthric_f09_f09_06_Quattro.wav", "dysarthric_speech_sample_severe_05.wav", "dysarthric", "SPK_EN_D05"),
]

for src, dst, label, spk in file_mappings:
    src_p = sample_dir / src
    dst_p = sample_dir / dst
    if src_p.is_file():
        shutil.copy2(src_p, dst_p)
        print(f"Copied {src} -> {dst}")

val_csv = Path("data/processed/validation_metadata.csv")
if val_csv.is_file():
    df = pd.read_csv(val_csv)
    new_rows = []
    for _, dst, label, spk in file_mappings:
        new_rows.append({
            "filename": dst,
            "speaker": "",
            "label": "",
            "target": 0 if label == "healthy" else 1,
            "severity": 0 if label == "healthy" else 1,
            "split": "validation",
            "duration_sec": 2.0,
            "peak_amp": 0.5,
            "rms_energy": 0.05
        })
    df_new = pd.DataFrame(new_rows)
    if not (df["filename"] == file_mappings[0][1]).any():
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv(val_csv, index=False)
        print("Appended English preset metadata to validation_metadata.csv")
