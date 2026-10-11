# Dataset Report: EasyCall Dysarthric Speech Corpus

## 1. Overview & Source
- **Corpus Name:** EasyCall Corpus (Turrisi et al., INTERSPEECH 2021)
- **Repository:** Hugging Face `changelinglab/easycall-dysarthria`
- **Official Host:** Italian Institute of Technology (IIT), University of Ferrara, and Sant'Anna Hospital of Ferrara
- **License:** Creative Commons Attribution-NonCommercial 2.0 (CC BY-NC 2.0)
- **Labels:** healthy controls and dysarthric speakers, with severity metadata for dysarthric speakers where available.

## 2. Dataset Dimensions & Speaker Independence
- **Total Audio Recordings:** 21386
- **Total Independent Speakers:** 55
  - **Healthy Control Speakers:** 24 speakers (10077 recordings)
  - **Dysarthric Speakers:** 31 speakers (11309 recordings)
- **Gender & Pathology Separation:**
  - Female Controls (`fc*`): 10 speakers
  - Male Controls (`mc*`): 14 speakers
  - Female Dysarthric (`f*`): 11 speakers
  - Male Dysarthric (`m*`): 20 speakers

## 3. Provided Speaker-Independent Partitions
- **Training:** 11,901 catalogued recordings from 31 speakers.
- **Validation:** 4,272 catalogued recordings from 11 speakers.
- **Test:** 5,213 catalogued recordings from 13 speakers.
- **Speaker overlap:** none across the saved training, validation, and test feature arrays.

The existing feature exports contain 21,268 recordings (11,881 training, 4,237
validation, and 5,150 test). The initial feature-extraction run did not retain
per-record failure reasons; 118 catalogued recordings are absent from those
exports. Run `scripts/audit_feature_exports.py` to list the absent recordings.
Future runs of `scripts/extract_features.py` save per-record errors to
`data/metadata/feature_extraction_failures.csv`.

## 4. Acoustic Properties
- **Format:** Linear PCM mono audio embedded in Parquet containers.
- **Sample Rate:** 8,000 Hz native (resampled to 16,000 Hz in feature extraction pipeline).
- **Channels:** Single-channel mono.
- **Utterances:** Voice commands and phonetically balanced target words for smartphone control.

## 5. Limitations
- EasyCall uses Italian smartphone voice commands; results should not be assumed to generalize to other languages or recording conditions.
- The number of independent speakers remains limited, despite the number of utterances.
- This corpus and model are not a substitute for clinical evaluation.
