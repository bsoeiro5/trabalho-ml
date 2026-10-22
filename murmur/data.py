"""Data: metadata of the CirCor DigiScope dataset, labels, subject-level splits and audio pre-processing."""
import hashlib
import json
import os
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm import tqdm

CLASSES = ["Present", "Unknown", "Absent"]  # same order as the Challenge
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
WA_WEIGHTS = np.array([5.0, 3.0, 1.0])  # class weights of the weighted accuracy


# ----------------------------------------------------------------------------- metadata

def recordings_dir(data_dir):
    """`training_data/` inside the PhysioNet folder, or the folder itself (e.g. hidden data)."""
    data_dir = Path(data_dir)
    sub = data_dir / "training_data"
    return sub if sub.is_dir() else data_dir


def _patient_files(rec_dir):
    return sorted(p for p in Path(rec_dir).glob("*.txt") if p.stem.isdigit())


def parse_patient_file(path):
    """Parse `<patient>.txt`: a header line, one line per recording, then `#Key: value` fields."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    patient_id = int(lines[0].split()[0])
    recordings, fields = [], {}
    for line in lines[1:]:
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            fields[key.strip()] = value.strip()
        else:
            parts = line.split()
            wav = next(p for p in parts if p.endswith(".wav"))
            recordings.append({"location": parts[0], "wav": wav})
    return patient_id, recordings, fields


def _normalise_column(name):
    return name.strip().rstrip(":").strip().lower().replace(" ", "_")


def load_patients(data_dir):
    """One row per patient, read from `training_data.csv` (or from the `.txt` files).

    Columns are renamed to snake_case: patient_id, murmur, murmur_locations,
    campaign, additional_id, ...
    """
    csv = Path(data_dir) / "training_data.csv"
    if csv.exists():
        raw = pd.read_csv(csv)
    else:
        rows = []
        for txt in _patient_files(recordings_dir(data_dir)):
            pid, _, fields = parse_patient_file(txt)
            rows.append({"Patient ID": pid, **fields})
        raw = pd.DataFrame(rows).replace("nan", np.nan)
    df = raw.rename(columns=_normalise_column)
    df["patient_id"] = df["patient_id"].astype(int)
    if "additional_id" in df:
        df["additional_id"] = pd.to_numeric(df["additional_id"], errors="coerce")
    return df


def load_recordings(data_dir, require_files=True):
    """One row per recording: rec_id, patient_id, location, wav_path.

    Recordings are listed from the patient `.txt` files; without them, patient and
    location are parsed from file names like `<patient>_<location>[_<n>].wav`.
    """
    rec_dir = recordings_dir(data_dir)
    rows = []
    txt_files = _patient_files(rec_dir)
    if txt_files:
        for txt in txt_files:
            pid, recs, _ = parse_patient_file(txt)
            for r in recs:
                rows.append({"patient_id": pid, "location": r["location"], "wav_path": str(rec_dir / r["wav"])})
    else:
        for wav in sorted(rec_dir.glob("*.wav")):
            pid, location = wav.stem.split("_")[:2]
            rows.append({"patient_id": int(pid), "location": location, "wav_path": str(wav)})
    df = pd.DataFrame(rows, columns=["patient_id", "location", "wav_path"])
    if require_files:
        df = df.loc[df["wav_path"].map(os.path.exists).astype(bool)]
    df.insert(0, "rec_id", [Path(p).stem for p in df["wav_path"]])
    return df.reset_index(drop=True)


def subject_ids(patients):
    """Map each patient to a subject id.

    About 70 children were recorded in both campaigns (CC2014 and CC2015) and appear
    twice, with different patient ids linked by the `Additional ID` field. They must
    stay in the same split, so they share one subject id (the smallest patient id).
    """
    parent = {pid: pid for pid in patients["patient_id"]}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    if "additional_id" in patients:
        for pid, other in zip(patients["patient_id"], patients["additional_id"]):
            if pd.notna(other) and int(other) in parent:
                ra, rb = find(pid), find(int(other))
                parent[max(ra, rb)] = min(ra, rb)
    return patients["patient_id"].map(find).to_numpy()


def recording_labels(recordings, patients, mode="location"):
    """Label of each recording, derived from the label of its patient.

    mode="patient":  every recording gets the patient label.
    mode="location": recordings of a Present patient are Present only if the murmur
                     was heard at their location (field `Murmur locations`), else Absent.
    """
    if mode not in ("location", "patient"):
        raise ValueError(f"unknown label mode {mode!r}")
    pat = patients.set_index("patient_id")
    labels = []
    for pid, loc in zip(recordings["patient_id"], recordings["location"]):
        label = pat.at[pid, "murmur"]
        locs = pat.at[pid, "murmur_locations"]
        if mode == "location" and label == "Present" and isinstance(locs, str):
            label = "Present" if loc in locs.split("+") else "Absent"
        labels.append(CLASS_TO_IDX[label])
    return np.array(labels, dtype=int)


# ----------------------------------------------------------------------------- splits

def _group_folds(strata, groups, n_folds, seed):
    x = np.zeros(len(strata))
    with warnings.catch_warnings():  # tiny strata (e.g. Unknown x CC2014) trigger a warning
        warnings.simplefilter("ignore", UserWarning)
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return [val for _, val in sgkf.split(x, strata, groups)]
        except ValueError:  # every stratum smaller than n_folds (only on tiny datasets): grouped only
            return [val for _, val in GroupKFold(n_splits=n_folds).split(x, strata, groups)]


def make_splits(patients, n_folds=5, test_fraction=0.2, seed=42):
    """Hold out a test set, then split the remaining (dev) patients into CV folds.

    Splits are grouped by subject (a child recorded in both campaigns has two patient
    ids) and stratified by murmur label x campaign. Columns of the result:
      subset    dev / test
      fold      CV fold of the dev patients (-1 for test patients)
      fold_all  folds over all patients, only used to retrain the final model on
                everything once the test evaluation is done
    """
    patients = patients.reset_index(drop=True)
    df = patients[["patient_id", "murmur", "campaign"]].copy()
    df["subject_id"] = subject_ids(patients)
    strata = (df["murmur"] + "|" + df["campaign"].fillna("")).to_numpy()
    groups = df["subject_id"].to_numpy()

    test_idx = _group_folds(strata, groups, int(round(1 / test_fraction)), seed)[0]
    df["subset"] = "dev"
    df.loc[test_idx, "subset"] = "test"

    df["fold"] = -1
    df["fold_all"] = -1
    for subset, columns in [("dev", ["fold", "fold_all"]), ("test", ["fold_all"])]:
        idx = np.flatnonzero(df["subset"] == subset)
        for k, val in enumerate(_group_folds(strata[idx], groups[idx], n_folds, seed)):
            df.loc[idx[val], columns] = k
    return df


# ----------------------------------------------------------------------------- audio

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
