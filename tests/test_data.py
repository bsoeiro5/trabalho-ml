import numpy as np
import pandas as pd

from murmur.data import parse_patient_file, recording_labels, subject_ids
from murmur.splits import make_splits

TXT = """49748 3 4000
AV 49748_AV_1.hea 49748_AV_1.wav 49748_AV_1.tsv
AV 49748_AV_2.hea 49748_AV_2.wav 49748_AV_2.tsv
MV 49748_MV.hea 49748_MV.wav 49748_MV.tsv
#Murmur: Present
#Murmur locations: AV+PV+TV
"""


def test_parse_patient_file(tmp_path):
    path = tmp_path / "49748.txt"
    path.write_text(TXT)
    pid, recs, fields = parse_patient_file(path)
    assert pid == 49748 and len(recs) == 3
    assert recs[1] == {"location": "AV", "wav": "49748_AV_2.wav"}
    assert fields["Murmur locations"] == "AV+PV+TV"


def test_recording_labels():
    patients = pd.DataFrame({"patient_id": [1, 2, 3], "murmur": ["Present", "Unknown", "Present"],
                             "murmur_locations": ["AV+PV", np.nan, np.nan]})
    recs = pd.DataFrame({"patient_id": [1, 1, 2, 3], "location": ["AV", "MV", "AV", "TV"]})
    assert recording_labels(recs, patients, "location").tolist() == [0, 2, 1, 0]
    assert recording_labels(recs, patients, "patient").tolist() == [0, 0, 1, 0]


def test_subject_ids_link_both_campaigns():
    patients = pd.DataFrame({"patient_id": [10, 20, 30], "additional_id": [30, np.nan, 10]})
    assert subject_ids(patients).tolist() == [10, 20, 10]


def test_splits_keep_subjects_together():
    rng = np.random.default_rng(0)
    n = 300
    ids = np.arange(1000, 1000 + n)
    add = np.full(n, np.nan)
    add[:40:2], add[1:40:2] = ids[1:40:2], ids[:40:2]
    patients = pd.DataFrame({"patient_id": ids, "additional_id": add,
                             "murmur": rng.choice(["Present", "Unknown", "Absent"], n, p=[0.2, 0.1, 0.7]),
                             "campaign": rng.choice(["CC2014", "CC2015"], n)})
    s = make_splits(patients)
    assert s.groupby("subject_id")["subset"].nunique().max() == 1
    assert s.groupby("subject_id")["fold_all"].nunique().max() == 1
    assert abs((s["subset"] == "test").mean() - 0.2) < 0.05
    assert set(s.loc[s["subset"] == "dev", "fold"]) == set(range(5))
