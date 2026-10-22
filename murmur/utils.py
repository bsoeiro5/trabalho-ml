"""Utilities: YAML configuration, seeding, device selection, JSON and logging."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# ----------------------------------------------------------------------------- configuration
#
# Experiment configuration: YAML files merged in order, plus `key.sub=value` overrides.
# A config file may start with `inherit: other.yaml` (path relative to the file) to
# extend another config.

def _merge(base, new):
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    return _merge(_read(path.parent / parent), cfg)


def _parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML reads "1e-3" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_override(cfg, item):
    """Apply an override such as `train.lr=3e-4` (the value is parsed as YAML)."""
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"override must look like key.sub=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = _parse_value(value)


def load_config(paths, overrides=()):
    cfg = {}
    for p in paths:
        _merge(cfg, _read(p))
    for item in overrides:
        apply_override(cfg, item)
    return cfg


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ----------------------------------------------------------------------------- misc

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
