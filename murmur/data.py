"""Metadata of the CirCor DigiScope dataset: patients, recordings and labels."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Present", "Unknown", "Absent"]  # same order as the Challenge
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
WA_WEIGHTS = np.array([5.0, 3.0, 1.0])  # class weights of the weighted accuracy


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
    pat = patients.set_index("patient_id")
    labels = []
    for pid, loc in zip(recordings["patient_id"], recordings["location"]):
        label = pat.at[pid, "murmur"]
        locs = pat.at[pid, "murmur_locations"]
        if mode == "location" and label == "Present" and isinstance(locs, str):
            label = "Present" if loc in locs.split("+") else "Absent"
        elif mode not in ("location", "patient"):
            raise ValueError(f"unknown label mode {mode!r}")
        labels.append(CLASS_TO_IDX[label])
    return np.array(labels, dtype=int)
