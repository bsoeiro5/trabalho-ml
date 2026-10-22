import numpy as np
import torch

from murmur.config import load_config
from murmur.engine import make_windows, window_starts
from murmur.features import deltas, mel_filterbank
from murmur.models import build_model


def test_models_forward():
    for name in ("cnn", "rnn"):
        cfg = load_config([f"configs/{name}.yaml"])
        model = build_model(cfg)
        x = torch.randn(3, int(cfg["windows"]["length_s"] * cfg["audio"]["fs"]))
        assert model(x).shape == (3, 3)
        model.eval()
        assert model(x[:, :5000]).shape == (3, 3)  # any length works


def test_rnn_attention_sums_to_one():
    cfg = load_config(["configs/rnn.yaml"])
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
    import pandas as pd

    from murmur.interpret import state_per_frame, weight_by_state
    seg = pd.DataFrame({"start": [0.0, 0.5, 1.0], "end": [0.5, 1.0, 1.2], "state": [1, 2, 3]})
    states = state_per_frame(seg, 0.25, 6, 0.25)  # t = 0.25 0.5 0.75 1.0 1.25 1.5
    assert states.tolist() == [1, 2, 2, 3, 0, 0]
    shares = weight_by_state(np.array([0, 1, 1, 0, 0, 0]), states)
    assert np.isclose(shares.loc["systole", "weight_share"], 1.0)
    assert np.isclose(shares.loc["systole", "time_share"], 2 / 6)
