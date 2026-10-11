"""
feature_extraction.py
Extracts acoustic speech summary features:
- MFCC (20 coefficients) + Delta + Delta-Delta statistics
- Spectral characteristics (Centroid, Bandwidth, Rolloff, Flatness, Contrast)
- Prosodic & Energy stability (RMS, Zero-Crossing Rate)

Outputs a deterministic, 144-dimensional acoustic feature vector.
Also provides log-Mel spectrogram generation for frontend visual display.
"""
import numpy as np
import librosa
from scipy import stats

FEATURE_DIM = 144
N_MFCC = 20
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

def extract_acoustic_features(audio_signal, sr=16000):
    """
    Extracts a fixed 144-dimensional acoustic vector from 1D audio_signal.
    
    Features breakdown:
      - MFCC (20): mean (20), std (20)                                -> 40
      - Delta MFCC (20): mean (20), std (20)                          -> 40
      - Delta-Delta MFCC (20): mean (20), std (20)                    -> 40
      - Spectral Centroid: mean, std                                   -> 2
      - Spectral Bandwidth: mean, std                                  -> 2
      - Spectral Rolloff (85%): mean, std                              -> 2
      - Spectral Flatness: mean, std                                   -> 2
      - Spectral Contrast (6 bands): mean (6), std (6)                 -> 12
      - RMS Energy: mean, std                                          -> 2
      - Zero Crossing Rate: mean, std                                  -> 2
    Total: 40 + 40 + 40 + 2 + 2 + 2 + 2 + 12 + 2 + 2 = 144
    """
    if len(audio_signal) < N_FFT:
        # Pad short signals to minimum length for STFT
        audio_signal = np.pad(audio_signal, (0, N_FFT - len(audio_signal)), mode='constant')

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio_signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 2. Delta & Delta-Delta MFCCs
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    # 3. Spectral Features
    spec_cent = librosa.feature.spectral_centroid(y=audio_signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    cent_mean = np.mean(spec_cent)
    cent_std = np.std(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bw_mean = np.mean(spec_bw)
    bw_std = np.std(spec_bw)

    spec_roll = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)
    roll_mean = np.mean(spec_roll)
    roll_std = np.std(spec_roll)

    spec_flat = librosa.feature.spectral_flatness(y=audio_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flat_mean = np.mean(spec_flat)
    flat_std = np.std(spec_flat)

    spec_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=5)
    contrast_mean = np.mean(spec_contrast, axis=1) # shape (6,)
    contrast_std = np.std(spec_contrast, axis=1)   # shape (6,)

    # 4. Energy and Phonation stability
    rms = librosa.feature.rms(y=audio_signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y=audio_signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Concatenate all features into a 1D vector
    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        [cent_mean, cent_std],
        [bw_mean, bw_std],
        [roll_mean, roll_std],
        [flat_mean, flat_std],
        contrast_mean, contrast_std,
        [rms_mean, rms_std],
        [zcr_mean, zcr_std]
    ]).astype(np.float32)

    assert len(feature_vector) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(feature_vector)}"

    # Replace any non-finite numbers with 0.0
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector

def compute_log_mel_spectrogram(audio_signal, sr=16000, n_mels=N_MELS, max_frames=300):
    """
    Computes a 2D Log-Mel spectrogram representation for audio visualization and analysis.
    Returns:
      log_mel: np.ndarray, shape (n_mels, time_steps), normalized to [0, 1] range for visual rendering.
    """
    if len(audio_signal) < N_FFT:
        audio_signal = np.pad(audio_signal, (0, N_FFT - len(audio_signal)), mode='constant')

    mel_spec = librosa.feature.melspectrogram(
        y=audio_signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Clip or pad time dimension if max_frames is set
    if max_frames is not None:
        if log_mel.shape[1] > max_frames:
            log_mel = log_mel[:, :max_frames]
        elif log_mel.shape[1] < max_frames:
            pad_width = max_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
            
    # Normalize to [0, 1] for visual display
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val > min_val:
        norm_mel = (log_mel - min_val) / (max_val - min_val)
    else:
        norm_mel = np.zeros_like(log_mel)

    return norm_mel
