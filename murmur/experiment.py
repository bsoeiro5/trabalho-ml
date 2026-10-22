"""Shared helpers to load the data of an experiment and the models/predictions it produced."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .aggregation import PROB_COLS, fit_decision
from .config import load_config
from .data import CLASS_TO_IDX, load_patients, load_recordings, recording_labels
from .engine import predict_recordings
from .models import build_model
from .utils import load_json


def load_dataset(cfg):
    """Patients, recordings (with split columns and recording labels) and patient labels."""
    data_dir = cfg["data"]["dir"]
    patients = load_patients(data_dir)
    splits = pd.read_csv(cfg["data"]["splits"])
    recordings = load_recordings(data_dir).merge(
        splits[["patient_id", "subset", "fold", "fold_all"]], on="patient_id", how="inner")
    recordings["label"] = recording_labels(recordings, patients, cfg["data"]["label_mode"])
    patient_labels = patients.set_index("patient_id")["murmur"].map(CLASS_TO_IDX)
    return patients, recordings, patient_labels, splits


def experiment_dir(cfg_or_dir, name=None):
    if isinstance(cfg_or_dir, dict):
        return Path(cfg_or_dir["output_dir"]) / name
    return Path(cfg_or_dir)


def load_experiment(exp_dir):
    """Config, CV summary and out-of-fold recording predictions of a finished CV run."""
    exp_dir = Path(exp_dir)
    cfg = load_config([exp_dir / "config.yaml"])
    summary = load_json(exp_dir / "cv_summary.json")
    oof = pd.read_csv(exp_dir / "oof_recordings.csv")
    return cfg, summary, oof


def combine_oof(oofs):
    """Average the out-of-fold probabilities of several experiments (same recordings and folds)."""
    base = oofs[0][["rec_id", "patient_id", "location", "fold"]].sort_values("rec_id").reset_index(drop=True)
    probs = []
    for oof in oofs:
        o = oof.set_index("rec_id").loc[base["rec_id"]]
        if not (o["fold"].to_numpy() == base["fold"].to_numpy()).all():
            raise ValueError("experiments were trained on different folds and cannot be combined")
        probs.append(o[PROB_COLS].to_numpy())
    base[PROB_COLS] = np.mean(probs, axis=0)
    return base


def decision_for(oof, patient_labels, methods):
    """Decision rule (aggregation + multipliers) fitted on out-of-fold predictions."""
    folds = oof.groupby("patient_id")["fold"].first()
    return fit_decision(oof, patient_labels, folds, methods)


def fold_models(exp_dir, device):
    """Yield (fold, model) for every fold model saved in an experiment directory."""
    for path in sorted(Path(exp_dir).glob("fold*/model.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        yield int(path.parent.name[4:]), model.to(device).eval()


def predict_with_experiment(exp_dir, cfg, signals_list, device):
    """Recording probabilities of every fold model of an experiment: array (n_folds, R, 3)."""
    fs = cfg["audio"]["fs"]
    length, hop = int(cfg["windows"]["length_s"] * fs), int(cfg["windows"]["eval_hop_s"] * fs)
    return np.stack([predict_recordings(m, signals_list, length, hop, device) for _, m in fold_models(exp_dir, device)])
