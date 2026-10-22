"""Training and inference loops."""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .aggregation import PROB_COLS, decide, patient_scores
from .data import WA_WEIGHTS
from .metrics import weighted_accuracy
from .models import build_model, count_parameters


class CropDataset(Dataset):
    """Random fixed-length crops of the training recordings (new crops every epoch).

    Each recording appears `crops_per_recording` times per epoch, so long recordings
    do not dominate. With probability `speed_p` the crop is time-stretched by a factor
    in `speed`, which simulates a different heart rate.
    """

    def __init__(self, signals, labels, length, crops_per_recording=4, speed=None, speed_p=0.0, seed=0):
        self.signals, self.labels = signals, labels
        self.length, self.crops = length, crops_per_recording
        self.speed, self.speed_p = speed, speed_p
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.signals) * self.crops

    def __getitem__(self, i):
        i %= len(self.signals)
        x = self.signals[i]
        factor = self.rng.uniform(*self.speed) if self.speed and self.rng.random() < self.speed_p else 1.0
        n = int(round(self.length * factor))
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        start = self.rng.integers(0, len(x) - n + 1)
        crop = x[start:start + n].astype(np.float32)
        if n != self.length:
            crop = np.interp(np.linspace(0, n - 1, self.length), np.arange(n), crop).astype(np.float32)
        return torch.from_numpy(crop), int(self.labels[i])


def augment_waves(x, gain_db=0.0, noise_snr_db=None, noise_p=0.0):
    """Random gain and additive white noise at a random SNR, applied to a batch (B, N)."""
    B = x.shape[0]
    if gain_db:
        g = (torch.rand(B, 1, device=x.device) * 2 - 1) * gain_db
        x = x * 10 ** (g / 20)
    if noise_snr_db and noise_p > 0:
        lo, hi = noise_snr_db
        snr = lo + (hi - lo) * torch.rand(B, 1, device=x.device)
        power = x.pow(2).mean(dim=1, keepdim=True)
        noise = torch.randn_like(x) * torch.sqrt(power / 10 ** (snr / 10))
        x = x + (torch.rand(B, 1, device=x.device) < noise_p).float() * noise
    return x


def window_starts(n, length, hop):
    """Start indices of sliding windows covering the whole signal (the last one ends at the end)."""
    if n <= length:
        return [0]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


def make_windows(signals, length, hop):
    """Cut every signal into windows; returns windows (W, length) and the recording of each window."""
    windows, owner = [], []
    for r, x in enumerate(signals):
        if len(x) < length:
            x = np.pad(x, (0, length - len(x)))
        for s in window_starts(len(x), length, hop):
            windows.append(x[s:s + length])
            owner.append(r)
    return np.stack(windows), np.array(owner)


@torch.no_grad()
def predict_windows(model, windows, device, batch_size=256):
    model.eval()
    out = []
    for i in range(0, len(windows), batch_size):
        xb = torch.from_numpy(windows[i:i + batch_size].astype(np.float32)).to(device)
        out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def mean_by_owner(values, owner, n):
    out = np.zeros((n, values.shape[1]))
    np.add.at(out, owner, values)
    return out / np.bincount(owner, minlength=n)[:, None]


def predict_recordings(model, signals, length, hop, device, batch_size=256):
    """Probabilities of each recording = mean of the probabilities of its windows, shape (R, 3)."""
    windows, owner = make_windows(signals, length, hop)
    return mean_by_owner(predict_windows(model, windows, device, batch_size), owner, len(signals))


def class_weights(spec, labels, n_classes=3):
    """Loss weights: 'challenge' (5, 3, 1, the metric weights), 'balanced', 'none' or an explicit list."""
    if spec == "challenge":
        w = WA_WEIGHTS
    elif spec == "balanced":
        counts = np.bincount(labels, minlength=n_classes)
        w = len(labels) / (n_classes * np.maximum(counts, 1))
    elif spec == "none":
        w = np.ones(n_classes)
    else:
        w = np.asarray(spec, dtype=float)
    return torch.tensor(w, dtype=torch.float32)


def patient_wa(val_rec, probs, patient_labels, method):
    rec = val_rec[["patient_id"]].copy()
    rec[PROB_COLS] = probs
    scores = patient_scores(rec, method)
    return weighted_accuracy(patient_labels.loc[scores.index].to_numpy(), decide(scores.to_numpy()))


def train_model(cfg, train_signals, train_labels, val_rec, val_signals, patient_labels, device, log=print):
    """Train one model, with early stopping on the patient-level WA of the validation patients.

    train_signals / train_labels: lists of training recordings and their labels.
    val_rec: DataFrame of the validation recordings (patient_id, label), aligned with val_signals.
    patient_labels: Series patient_id -> class index.
    Returns the model with the best validation weights and the per-epoch history.
    """
    tcfg, aug = cfg["train"], cfg["train"].get("augment", {})
    fs = cfg["audio"]["fs"]
    length = int(cfg["windows"]["length_s"] * fs)
    hop = int(cfg["windows"]["eval_hop_s"] * fs)

    model = build_model(cfg).to(device)
    log(f"  model: {count_parameters(model):,} parameters")
    dataset = CropDataset(train_signals, train_labels, length, tcfg["crops_per_recording"],
                          aug.get("speed"), aug.get("speed_p", 0.0), seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True,
                        drop_last=len(dataset) > tcfg["batch_size"])
    weights = class_weights(tcfg["class_weights"], np.asarray(train_labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps, warmup = tcfg["epochs"] * len(loader), len(loader)  # 1 epoch of linear warm-up, then cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    val_windows, val_owner = make_windows(val_signals, length, hop)
    val_window_labels = torch.as_tensor(val_rec["label"].to_numpy()[val_owner])
    monitor = cfg["decision"]["monitor"]
    best_key, best_state, bad, history = None, None, 0, []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running, n_batches = torch.zeros((), device=device), 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment_waves(xb, aug.get("gain_db", 0.0), aug.get("noise_snr_db"), aug.get("noise_p", 0.0))
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 5.0))
            optimizer.step()
            scheduler.step()
            running += loss.detach()
            n_batches += 1

        p_win = predict_windows(model, val_windows, device)
        val_loss = F.nll_loss(torch.log(torch.as_tensor(p_win) + 1e-8), val_window_labels, weight=weights.cpu()).item()
        rec_probs = mean_by_owner(p_win, val_owner, len(val_signals))
        val_wa = patient_wa(val_rec, rec_probs, patient_labels, monitor)
        row = {"epoch": epoch, "train_loss": (running / max(n_batches, 1)).item(), "val_loss": val_loss,
               "val_wa": val_wa, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0}
        history.append(row)

        key = (val_wa, -val_loss)  # best WA, ties broken by the validation loss
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_state, bad = key, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), 0
        else:
            bad += 1
        log(f"  epoch {epoch:3d}  train_loss {row['train_loss']:.4f}  val_loss {val_loss:.4f}  "
            f"val_WA {val_wa:.4f}{' *' if improved else ''}  ({row['time_s']:.1f}s)")
        if bad >= tcfg["patience"]:
            log(f"  early stopping (no improvement for {bad} epochs)")
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
