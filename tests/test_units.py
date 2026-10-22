"""Unit tests: metric, labels and splits, decision rule, features and models."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from murmur.data import make_splits, parse_patient_file, recording_labels, subject_ids
from murmur.evaluation import (classification_metrics, confusion_matrix, crossfit_decision, decide, patient_scores,
                               tune_scale, weighted_accuracy)
from murmur.features import deltas, mel_filterbank
from murmur.interpret import state_per_frame, weight_by_state
from murmur.models import build_model
from murmur.training import EMA, make_windows, window_starts
from murmur.utils import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# ----------------------------------------------------------------------------- metric

def official_weighted_accuracy(labels, outputs):
    """Re-implementation of compute_weighted_accuracy from physionetchallenges/evaluation-2022
    (one-hot labels/outputs, classes ordered Present, Unknown, Absent)."""
    weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    A = np.zeros((3, 3))
    for i in range(len(labels)):
        A[np.argmax(outputs[i]), np.argmax(labels[i])] += 1  # rows = outputs, columns = labels
    return np.trace(weights * A) / np.sum(weights * A)


def test_matches_formula_and_official_code():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    m = confusion_matrix(y, p)
    formula = (5 * m[0, 0] + 3 * m[1, 1] + m[2, 2]) / (5 * m[:, 0].sum() + 3 * m[:, 1].sum() + m[:, 2].sum())
    assert np.isclose(weighted_accuracy(y, p), formula)
    assert np.isclose(weighted_accuracy(y, p), official_weighted_accuracy(np.eye(3)[y], np.eye(3)[p]))


def test_trivial_classifiers_on_public_data():
    y = np.array([0] * 179 + [1] * 68 + [2] * 695)  # public CirCor: Present, Unknown, Absent
    assert np.isclose(weighted_accuracy(y, np.full_like(y, 2)), 695 / 1794)  # always Absent: 0.387
    assert np.isclose(weighted_accuracy(y, np.full_like(y, 0)), 895 / 1794)  # always Present: 0.499


def test_confusion_matrix_convention():
    m = confusion_matrix(np.array([0, 0, 2]), np.array([2, 0, 2]))  # rows = predicted, columns = true
    assert m[2, 0] == 1 and m[0, 0] == 1 and m[2, 2] == 1
    r = classification_metrics(np.array([0, 0, 2]), np.array([2, 0, 2]))
    assert np.isclose(r["recall_Present"], 0.5) and np.isclose(r["precision_Absent"], 0.5)


# ----------------------------------------------------------------------------- data: patient files, labels, subjects, splits

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


# ----------------------------------------------------------------------------- decision rule

def test_patient_scores():
    rec = pd.DataFrame({"patient_id": [1, 1, 2], "p_present": [0.9, 0.1, 0.2],
                        "p_unknown": [0.0, 0.1, 0.1], "p_absent": [0.1, 0.8, 0.7]})
    mean, mx = patient_scores(rec, "mean"), patient_scores(rec, "max")
    assert np.isclose(mean.loc[1, "p_present"], 0.5) and np.isclose(mx.loc[1, "p_present"], 0.9)
    assert decide(mx.to_numpy()).tolist() == [0, 2]


def test_tuned_scale_prefers_argmax_when_it_is_optimal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    scores = np.eye(3)[y] * 0.8 + 0.1  # perfectly separable: argmax is optimal
    assert np.allclose(tune_scale(scores, y), [1, 1, 1])


def test_crossfit_improves_biased_scores():
    rng = np.random.default_rng(1)
    y = rng.choice(3, 600, p=[0.2, 0.1, 0.7])
    scores = rng.dirichlet(np.ones(3), 600) * 0.5 + np.eye(3)[y] * 0.5
    scores[:, 0] *= 0.3  # the model under-predicts Present
    res = crossfit_decision(scores, y, np.arange(600) % 5)
    assert np.mean(res["per_fold"]["tuned"]) > np.mean(res["per_fold"]["argmax"])


# ----------------------------------------------------------------------------- features and models

def test_models_forward():
    for name in ("cnn", "rnn"):
        cfg = load_config([CONFIGS / f"{name}.yaml"])
        model = build_model(cfg)
        x = torch.randn(3, int(cfg["windows"]["length_s"] * cfg["audio"]["fs"]))
        assert model(x).shape == (3, 3)
        model.eval()
        assert model(x[:, :5000]).shape == (3, 3)  # any length works


def test_rnn_attention_sums_to_one():
    cfg = load_config([CONFIGS / "rnn.yaml"])
    model = build_model(cfg).eval()
    _, att = model(torch.randn(2, 8000), return_attention=True)
    assert torch.allclose(att.sum(1), torch.ones(2))


def test_features():
    fb = mel_filterbank(2000, 256, 64, 25, 800)
    assert fb.shape == (64, 129) and (fb.sum(1) > 0).all()
    ramp = torch.arange(20.0).repeat(1, 3, 1)  # slope 1
    assert torch.allclose(deltas(ramp)[..., 2:-2], torch.ones(1, 3, 16))


def test_windows_cover_the_signal():
    assert window_starts(10, 4, 2) == [0, 2, 4, 6]
    assert window_starts(11, 4, 2) == [0, 2, 4, 6, 7]
    w, owner = make_windows([np.ones(11), np.ones(3)], 4, 2)
    assert w.shape == (6, 4) and owner.tolist() == [0, 0, 0, 0, 0, 1]


def test_segmentation_frames():
    seg = pd.DataFrame({"start": [0.0, 0.5, 1.0], "end": [0.5, 1.0, 1.2], "state": [1, 2, 3]})
    states = state_per_frame(seg, 0.25, 6, 0.25)  # t = 0.25 0.5 0.75 1.0 1.25 1.5
    assert states.tolist() == [1, 2, 2, 3, 0, 0]
    shares = weight_by_state(np.array([0, 1, 1, 0, 0, 0]), states)
    assert np.isclose(shares.loc["systole", "weight_share"], 1.0)
    assert np.isclose(shares.loc["systole", "time_share"], 2 / 4)  # unannotated frames are ignored
    assert np.isclose(shares.loc["systole", "ratio"], 2.0)


def test_ema_moves_towards_the_weights():
    model = torch.nn.Linear(2, 1)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for _ in range(200):
        ema.update(model)
    assert torch.allclose(ema.model.weight, torch.ones(1, 2), atol=1e-3)
    assert not ema.model.weight.requires_grad
