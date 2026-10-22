"""Where do the models look? RNN attention and CNN saliency, compared with the S1/systole/S2/diastole segmentation.

The `.tsv` files give, for each recording, the cardiac state over time
(0 unannotated, 1 S1, 2 systole, 3 S2, 4 diastole). If a model detects murmurs for
the right reason, its attention/saliency should concentrate in systole (almost all
murmurs in the dataset are systolic).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STATES = {0: "unannotated", 1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def load_segmentation(wav_path):
    tsv = Path(wav_path).with_suffix(".tsv")
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "state"])


def state_per_frame(seg, start_s, n_frames, hop_s):
    """Cardiac state at the centre of each feature frame of a window starting at `start_s` seconds."""
    t = start_s + np.arange(n_frames) * hop_s
    idx = np.searchsorted(seg["end"].to_numpy(), t, side="right")
    states = np.zeros(n_frames, dtype=int)
    valid = idx < len(seg)
    states[valid] = seg["state"].to_numpy()[idx[valid]]
    states[valid & (t < seg["start"].to_numpy()[np.minimum(idx, len(seg) - 1)])] = 0
    return states


@torch.no_grad()
def rnn_attention(model, wave):
    """Class probabilities and attention weights over time (n_frames,) of an RNN model for one window."""
    model.eval()
    logits, att = model(torch.as_tensor(wave, dtype=torch.float32)[None], return_attention=True)
    return torch.softmax(logits, 1)[0].numpy(), att[0].numpy()


def cnn_saliency(model, wave, target=0):
    """|gradient x input| of the `target` logit w.r.t. the log-mel features (F, T), for one window."""
    model.eval()
    feats = model.frontend(torch.as_tensor(wave, dtype=torch.float32)[None]).detach().requires_grad_(True)
    logits = model.backbone(feats)
    logits[0, target].backward()
    return torch.softmax(logits.detach(), 1)[0].numpy(), (feats.grad * feats.detach()).abs()[0].numpy()


def time_profile(model, wave, kind):
    """Probabilities and per-frame importance for the Present class:
    RNN attention, or CNN saliency summed over frequency."""
    if kind == "rnn":
        return rnn_attention(model, wave)
    probs, saliency = cnn_saliency(model, wave, target=0)
    return probs, saliency.sum(axis=0)


def attribution_by_state(model, kind, cfg, recordings, signals, max_windows=300):
    """Average share of importance per cardiac state over non-overlapping windows of `recordings`."""
    fs = cfg["audio"]["fs"]
    length = int(cfg["windows"]["length_s"] * fs)
    hop_s = cfg["features"]["hop_length"] / fs
    rows = []
    for rec in recordings.itertuples():
        seg = load_segmentation(rec.wav_path)
        if seg is None:
            continue
        x = signals[rec.rec_id].astype(np.float32)
        for s in range(0, max(len(x) - length, 0) + 1, length):
            wave = np.pad(x[s:s + length], (0, max(0, length - len(x[s:s + length]))))
            _, w = time_profile(model, wave, kind)
            rows.append(weight_by_state(w, state_per_frame(seg, s / fs, len(w), hop_s)))
            if len(rows) >= max_windows:
                return sum(rows) / len(rows), len(rows)
    return (sum(rows) / len(rows), len(rows)) if rows else (None, 0)


def weight_by_state(weights, states):
    """Share of the total weight (attention or saliency over time) falling in each cardiac state,
    next to the share of time spent in that state (the value expected from a uniform weighting)."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return pd.DataFrame({
        "weight_share": [weights[states == s].sum() / total for s in STATES],
        "time_share": [(states == s).mean() for s in STATES],
    }, index=list(STATES.values()))
