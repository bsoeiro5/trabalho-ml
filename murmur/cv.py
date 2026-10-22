"""Cross-validation on the development set: one model per fold + out-of-fold (OOF) predictions.

The test patients are never used here (unless --all-data). All decisions (architecture,
features, hyper-parameters, aggregation, decision multipliers) are taken from the CV results.

    python -m murmur.cv --config configs/cnn.yaml
    python -m murmur.cv --config configs/cnn.yaml --name cnn_lr3e-4 --set train.lr=3e-4
    python -m murmur.cv --config configs/cnn.yaml --folds 0 1          # quick exploration
    python -m murmur.cv --config configs/cnn.yaml --all-data           # final retraining (dev + test)
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from .aggregation import PROB_COLS, describe_decision
from .audio import load_signals
from .config import load_config, save_config
from .engine import predict_recordings, train_model
from .experiment import decision_for, load_dataset
from .utils import Logger, get_device, save_json, set_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--name", help="experiment name (default: `name` in the config)")
    parser.add_argument("--folds", nargs="*", type=int, help="run only these folds")
    parser.add_argument("--all-data", action="store_true",
                        help="CV over dev + test patients, to retrain the final model after the test evaluation")
    parser.add_argument("--resume", action="store_true", help="skip folds that are already finished")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    name = args.name or cfg["name"]
    if args.all_data and not name.endswith("_all"):
        name += "_all"
    out = Path(cfg["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    log = Logger(out / "train.log")
    device = get_device(args.device, cfg.get("device_preference", ("cuda", "mps", "cpu")))
    log(f"experiment {name} on {device}")

    patients, recordings, patient_labels, _ = load_dataset(cfg)
    fold_col = "fold_all" if args.all_data else "fold"
    pool = recordings if args.all_data else recordings[recordings["subset"] == "dev"]
    signals = load_signals(recordings, cfg["audio"], cfg.get("cache_dir"))
    fs = cfg["audio"]["fs"]
    length, hop = int(cfg["windows"]["length_s"] * fs), int(cfg["windows"]["eval_hop_s"] * fs)

    folds = args.folds if args.folds else sorted(int(k) for k in pool[fold_col].unique())
    oofs = []
    for k in folds:
        fold_dir = out / f"fold{k}"
        if args.resume and (fold_dir / "oof.csv").exists():
            log(f"fold {k}: already done, skipping")
            oofs.append(pd.read_csv(fold_dir / "oof.csv"))
            continue
        fold_dir.mkdir(exist_ok=True)
        set_seed(cfg["seed"] + k)
        tr, va = pool[pool[fold_col] != k], pool[pool[fold_col] == k].reset_index(drop=True)
        log(f"\nfold {k}: {tr['patient_id'].nunique()} train / {va['patient_id'].nunique()} validation patients, "
            f"{len(tr)} / {len(va)} recordings")
        t0 = time.time()
        val_signals = [signals[r] for r in va["rec_id"]]
        model, history = train_model(cfg, [signals[r] for r in tr["rec_id"]], tr["label"].to_numpy(),
                                     va, val_signals, patient_labels, device, log)
        torch.save({"state_dict": model.state_dict(), "config": cfg}, fold_dir / "model.pt")
        history.to_csv(fold_dir / "history.csv", index=False)

        oof = va[["rec_id", "patient_id", "location", "label"]].copy()
        oof["fold"] = k
        oof[PROB_COLS] = predict_recordings(model, val_signals, length, hop, device)
        oof.to_csv(fold_dir / "oof.csv", index=False)
        oofs.append(oof)
        log(f"fold {k} done in {(time.time() - t0) / 60:.1f} min (best val WA {history['val_wa'].max():.4f})")

    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(out / "oof_recordings.csv", index=False)
    decision = decision_for(oof, patient_labels, cfg["decision"]["methods"])
    summary = {"experiment": name, "all_data": args.all_data, "folds": [int(k) for k in folds],
               "decision": {k: v for k, v in decision.items() if k != "methods"},
               "methods": decision["methods"]}
    save_json(summary, out / "cv_summary.json")
    log(f"\n=== {name}: patient-level weighted accuracy over {len(folds)} folds ===\n{describe_decision(decision)}")


if __name__ == "__main__":
    main()
