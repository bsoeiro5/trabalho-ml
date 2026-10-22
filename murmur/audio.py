"""Loading and pre-processing of the heart sound recordings."""
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm


def load_wav(path):
    """Read a wav file as float32 in [-1, 1]; returns (signal, sampling rate)."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return x.astype(np.float32), fs


def preprocess(x, fs, target_fs=2000, band=(25.0, 800.0)):
    """Band-pass filter, resample to `target_fs` and standardise (zero mean, unit variance).

    The band keeps the heart sounds (S1/S2, mostly < 200 Hz) and the murmurs (up to
    ~600 Hz) while removing very low-frequency noise (movement, breathing) and
    high-frequency noise. Resampling 4000 -> 2000 Hz halves the cost without losing
    information inside the band.
    """
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    if fs != target_fs:
        g = gcd(int(fs), int(target_fs))
        x = resample_poly(x, int(target_fs) // g, int(fs) // g)
    x = x - x.mean()
    return (x / (x.std() + 1e-8)).astype(np.float32)


def load_signals(recordings, audio_cfg, cache_dir=None):
    """Pre-processed signal of every recording, as a dict rec_id -> float16 array.

    The result is cached on disk; the cache key depends on the files and on the
    pre-processing parameters, so changing either rebuilds it.
    """
    target_fs, band = audio_cfg["fs"], tuple(audio_cfg["band"])
    files = sorted(str(Path(p).resolve()) for p in recordings["wav_path"])
    key = json.dumps({"files": files, "fs": target_fs, "band": band})
    path = None
    if cache_dir:
        path = Path(cache_dir) / f"signals_{hashlib.md5(key.encode()).hexdigest()[:12]}.npz"
        if path.exists():
            with np.load(path) as f:
                return {k: f[k] for k in f.files}
    signals = {}
    items = zip(recordings["rec_id"], recordings["wav_path"])
    for rec_id, wav in tqdm(items, total=len(recordings), desc="pre-processing audio"):
        x, fs = load_wav(wav)
        signals[rec_id] = preprocess(x, fs, target_fs, band).astype(np.float16)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **signals)
    return signals
