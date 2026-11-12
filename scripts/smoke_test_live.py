import numpy as np
import io
import json
import urllib.request
import soundfile as sf

sr = 16000
t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
signal = (0.5*np.sin(2*np.pi*130*t) + 0.3*np.sin(2*np.pi*260*t) +
          0.2*np.sin(2*np.pi*500*t) + 0.15*np.sin(2*np.pi*1500*t))
signal = (signal / np.max(np.abs(signal)) * 0.65).astype(np.float32)

bio = io.BytesIO()
sf.write(bio, signal, sr, format="WAV")
wav_bytes = bio.getvalue()

boundary = b"NeuroVoiceBoundary7348"
body = (
    b"--" + boundary + b"\r\n"
    b"Content-Disposition: form-data; name=\"audio_file\"; filename=\"test_healthy.wav\"\r\n"
    b"Content-Type: audio/wav\r\n\r\n" +
    wav_bytes +
    b"\r\n--" + boundary + b"--\r\n"
)

req = urllib.request.Request(
    "http://localhost:5000/analyze",
    data=body,
    headers={
        "Content-Type": "multipart/form-data; boundary=NeuroVoiceBoundary7348",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json"
    }
)
with urllib.request.urlopen(req) as r:
    res = json.loads(r.read())
    print("Live Healthy Voice Test:")
    print("  Prediction   :", res["prediction"])
    print("  Healthy score:", res["score_healthy_pct"], "%")
    print("  Dysarthric   :", res["score_dysarthric_pct"], "%")
    print("  Model used   :", res["model_used"])
    print("  Success      :", res["success"])
