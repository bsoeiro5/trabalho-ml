"""Predict the murmur label of every patient in a folder of recordings (e.g. the hidden data of the bonus).

The folder may contain the patient `.txt` files (as in the public data) or only the
`.wav` files named `<patient>_<location>[_<n>].wav`. Output: one row per patient.

    python -m murmur.predict --exp cnn_all rnn_all --data-dir data/hidden --out predictions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import PROB_COLS, decide, describe_decision, patient_scores
from .audio import load_signals
from .data import CLASSES, load_recordings
from .evaluate import resolve
from .experiment import combine_oof, decision_for, load_dataset, load_experiment, predict_with_experiment
from .utils import get_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", required=True, help="experiment names (or directories)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    exp_dirs = [resolve(e, args.output_dir) for e in args.exp]
    exps = [load_experiment(d) for d in exp_dirs]
    cfg = exps[0][0]
    _, _, patient_labels, _ = load_dataset(cfg)
    decision = decision_for(combine_oof([o for _, _, o in exps]), patient_labels, cfg["decision"]["methods"])
    print(f"decision rule (fitted on the out-of-fold predictions):\n{describe_decision(decision)}")

    recordings = load_recordings(args.data_dir)
    if recordings.empty:
        raise SystemExit(f"no recordings found in {args.data_dir}")
    print(f"{len(recordings)} recordings of {recordings['patient_id'].nunique()} patients")
    device = get_device(args.device, cfg.get("device_preference", ("cuda", "mps", "cpu")))
    probs = []
    for (exp_cfg, _, _), d in zip(exps, exp_dirs):
        signals = load_signals(recordings, exp_cfg["audio"])
        probs.append(predict_with_experiment(d, exp_cfg, [signals[r] for r in recordings["rec_id"]], device).mean(axis=0))
    rec = recordings[["patient_id"]].copy()
    rec[PROB_COLS] = np.mean(probs, axis=0)
    scores = patient_scores(rec, decision["method"])
    pred = decide(scores.to_numpy(), decision["scale"])

    out = pd.DataFrame({"patient_id": scores.index, "murmur": [CLASSES[i] for i in pred]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out["murmur"].value_counts().to_string())
    print(f"predictions written to {args.out}")


if __name__ == "__main__":
    main()
