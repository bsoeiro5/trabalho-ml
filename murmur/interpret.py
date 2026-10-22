"""Where do the models look? RNN attention and CNN occlusion, compared with the S1/systole/S2/diastole segmentation.

The `.tsv` files give, for each recording, the cardiac state over time
(0 unannotated, 1 S1, 2 systole, 3 S2, 4 diastole). If a model detects murmurs for
the right reason, its attention/occlusion importance should concentrate in systole (almost all
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


def _batch(model, wave):
    """One window as a batch of 1, on the device of the model."""
    return torch.as_tensor(wave, dtype=torch.float32, device=next(model.parameters()).device)[None]


@torch.no_grad()
def rnn_attention(model, wave):
    """Class probabilities and attention weights over time (n_frames,) of an RNN model for one window."""
    model.eval()
    logits, att = model(_batch(model, wave), return_attention=True)
    return torch.softmax(logits, 1)[0].cpu().numpy(), att[0].cpu().numpy()


@torch.no_grad()
def occlusion_profile(model, wave, width=6, stride=3, target=0):
    """Class probabilities and importance over time (n_frames,) by occlusion, for one window.

    Every block of `width` feature frames (6 x 16 ms ~ 100 ms) is replaced by the mean of
    the window, and the importance of a frame is the average drop of P(target) when it is
    hidden (negative drops are set to 0). Unlike gradients, this is not dominated by the
    loudest sounds (S1/S2).
    """
    model.eval()
    feats = model.frontend(_batch(model, wave))
    p0 = torch.softmax(model.backbone(feats), 1)[0]
    T = feats.shape[-1]
    starts = list(range(0, T - width + 1, stride))
    batch = feats.repeat(len(starts), 1, 1)
    for i, s in enumerate(starts):
        batch[i, :, s:s + width] = feats.mean()
    drop = (p0[target] - torch.softmax(model.backbone(batch), 1)[:, target]).clamp(min=0).cpu().numpy()
    importance, count = np.zeros(T), np.zeros(T)
    for d, s in zip(drop, starts):
        importance[s:s + width] += d
        count[s:s + width] += 1
    return p0.cpu().numpy(), importance / np.maximum(count, 1)


def time_profile(model, wave, kind):
    """Probabilities and per-frame importance for the Present class: RNN attention, or CNN occlusion."""
    return rnn_attention(model, wave) if kind == "rnn" else occlusion_profile(model, wave)


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
            states = state_per_frame(seg, s / fs, len(w), hop_s)
            if (states != 0).mean() >= 0.5:  # skip windows that are mostly unannotated
                rows.append(weight_by_state(w, states))
            if len(rows) >= max_windows:
                break
        if len(rows) >= max_windows:
            break
    if not rows:
        return None, 0
    shares = sum(r[["weight_share", "time_share"]] for r in rows) / len(rows)
    shares["ratio"] = shares["weight_share"] / shares["time_share"]
    return shares, len(rows)


def weight_by_state(weights, states):
    """Share of the total weight (attention or occlusion importance over time) falling in each cardiac phase,
    the share of time spent in that phase (what a uniform weighting would give) and their ratio
    (> 1: the model looks at that phase more than its duration). Unannotated frames are ignored."""
    weights, states = np.asarray(weights, dtype=float), np.asarray(states)
    phases = {s: name for s, name in STATES.items() if s != 0}
    annotated = states != 0
    total, n = weights[annotated].sum(), annotated.sum()
    df = pd.DataFrame({
        "weight_share": [weights[states == s].sum() / total if total > 0 else np.nan for s in phases],
        "time_share": [(states == s).sum() / n if n > 0 else np.nan for s in phases],
    }, index=list(phases.values()))
    df["ratio"] = df["weight_share"] / df["time_share"]
    return df
