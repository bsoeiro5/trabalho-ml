"""Time-frequency features, computed on the fly (on the GPU) from the waveform.

Computing the features inside the model lets data augmentation act on the raw
waveform (random crops, time-stretching, noise) and keeps one single code path for
training, evaluation and prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(fs, n_fft, n_mels, fmin, fmax):
    """Triangular filters equally spaced on the mel scale, shape (n_mels, n_fft // 2 + 1)."""
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    lo, center, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = np.maximum(0.0, np.minimum((freqs - lo) / (center - lo), (hi - freqs) / (hi - center)))
    if (fb.sum(axis=1) == 0).any():
        raise ValueError("some mel filters are empty: use a larger n_fft or fewer mel bands")
    return torch.tensor(fb, dtype=torch.float32)


def dct_matrix(n_out, n_in):
    """Orthonormal DCT-II matrix (n_out, n_in), used to compute MFCCs from log-mel energies."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    m = np.cos(np.pi / n_in * (n + 0.5) * k) * np.sqrt(2.0 / n_in)
    m[0] /= np.sqrt(2.0)
    return torch.tensor(m, dtype=torch.float32)


def deltas(x, width=2):
    """Regression deltas along time of x (B, F, T): sum_n n x[t+n] / sum_n n^2."""
    n = torch.arange(-width, width + 1, dtype=x.dtype, device=x.device)
    kernel = (n / (n**2).sum()).repeat(x.shape[1], 1, 1)
    return F.conv1d(F.pad(x, (width, width), mode="replicate"), kernel, groups=x.shape[1])


class SpecAugment(nn.Module):
    """Random frequency and time masks (Park et al., 2019), only active in training."""

    def __init__(self, freq_masks=2, freq_width=8, time_masks=2, time_width=20):
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width

    def _mask(self, x, fill, dim, n_masks, max_width):
        B, size = x.shape[0], x.shape[dim]
        pos = torch.arange(size, device=x.device)[None]
        for _ in range(n_masks):
            width = torch.randint(0, max_width + 1, (B, 1), device=x.device)
            start = (torch.rand(B, 1, device=x.device) * (size - width).clamp(min=1)).long()
            mask = (pos >= start) & (pos < start + width)
            mask = mask[:, :, None] if dim == 1 else mask[:, None, :]
            x = torch.where(mask, fill, x)
        return x

    def forward(self, x):  # x: (B, F, T)
        if not self.training:
            return x
        fill = x.mean(dim=(1, 2), keepdim=True)
        x = self._mask(x, fill, 1, self.freq_masks, self.freq_width)
        return self._mask(x, fill, 2, self.time_masks, self.time_width)


class Frontend(nn.Module):
    """Waveform (B, N) -> features (B, F, T): log-mel spectrogram, or MFCCs (+ deltas)."""

    def __init__(self, fs, kind="logmel", n_fft=256, win_length=128, hop_length=32, n_mels=64,
                 fmin=25.0, fmax=800.0, n_mfcc=20, deltas=True, spec_augment=None):
        super().__init__()
        if kind not in ("logmel", "mfcc"):
            raise ValueError(f"unknown feature kind {kind!r}")
        self.kind, self.n_fft, self.win_length, self.hop_length = kind, n_fft, win_length, hop_length
        self.use_deltas = deltas
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("mel_fb", mel_filterbank(fs, n_fft, n_mels, fmin, fmax), persistent=False)
        if kind == "mfcc":
            self.register_buffer("dct", dct_matrix(n_mfcc, n_mels), persistent=False)
            self.n_features = n_mfcc * (3 if deltas else 1)
        else:
            self.n_features = n_mels
        self.spec_augment = SpecAugment(**spec_augment) if spec_augment else nn.Identity()

    def forward(self, wave):
        spec = torch.stft(wave, self.n_fft, self.hop_length, self.win_length, self.window,
                          return_complex=True).abs().pow(2)
        x = torch.log(self.mel_fb @ spec + 1e-5)  # (B, n_mels, T)
        x = self.spec_augment(x)
        if self.kind == "mfcc":
            x = self.dct @ x
            if self.use_deltas:
                d = deltas(x)
                x = torch.cat([x, d, deltas(d)], dim=1)
        return x
