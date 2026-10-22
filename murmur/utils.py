"""Small helpers: seeding, device selection, JSON and logging."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto", preference=("cuda", "mps", "cpu")):
    """`name`, or with "auto" the first available device in `preference`."""
    if name != "auto":
        return torch.device(name)
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    return torch.device(next(d for d in preference if available[d]))


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Logger:
    """Print to the terminal and append to a log file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None

    def __call__(self, msg):
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(msg + "\n")
