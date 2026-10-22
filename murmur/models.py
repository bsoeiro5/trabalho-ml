"""Neural network classifiers: a 2D CNN and a recurrent network (GRU/LSTM)."""
import torch
import torch.nn as nn

from .features import Frontend


class ConvBlock(nn.Module):
    """Two 3x3 convolutions (+ batch norm + ReLU), followed by 2x2 average pooling."""

    def __init__(self, c_in, c_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.AvgPool2d(2), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CNN2D(nn.Module):
    """VGG-like 2D CNN on a time-frequency representation (as in PANNs, Kong et al. 2020).

    Input (B, F, T). Each block halves both axes; at the end the frequency axis is
    averaged and the time axis is summarised by mean + max pooling, so the network
    accepts any window length.
    """

    def __init__(self, n_features, n_classes=3, channels=(16, 32, 64, 128), dropout=0.3, block_dropout=0.1):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_features)  # per-band normalisation of the input
        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out, block_dropout))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * c_in, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.blocks(self.bn0(x).unsqueeze(1))  # (B, C, F', T')
        x = x.mean(dim=2)                          # average over frequency
        x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.head(x)


class AttentionPooling(nn.Module):
    """Weighted average over time, with weights a_t = softmax_t(v^T tanh(W h_t))."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h):  # h: (B, T, D)
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(dim=1), a


class RNN(nn.Module):
    """(Bi)directional GRU/LSTM over the sequence of feature frames, with attention pooling.

    Input (B, F, T) is read as a sequence of T frames of F features. The attention
    weights show which instants (e.g. systole) drive the decision.
    """

    def __init__(self, n_features, n_classes=3, cell="gru", hidden=64, layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_features)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.pool = AttentionPooling(out_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(out_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attention=False):
        h, _ = self.rnn(self.bn0(x).transpose(1, 2))  # (B, T, D)
        z, a = self.pool(h)
        logits = self.head(z)
        return (logits, a) if return_attention else logits


class MurmurNet(nn.Module):
    """Complete model: waveform -> features (frontend) -> classifier (backbone) -> logits."""

    def __init__(self, frontend, backbone):
        super().__init__()
        self.frontend, self.backbone = frontend, backbone

    def forward(self, wave, **kwargs):
        return self.backbone(self.frontend(wave), **kwargs)


BACKBONES = {"cnn": CNN2D, "rnn": RNN}


def build_model(cfg):
    frontend = Frontend(fs=cfg["audio"]["fs"], **cfg["features"])
    kwargs = {k: v for k, v in cfg["model"].items() if k != "type"}
    backbone = BACKBONES[cfg["model"]["type"]](frontend.n_features, **kwargs)
    return MurmurNet(frontend, backbone)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
