"""Experiment pipeline, run as `python -m murmur <command>`.

  prepare   subject-level splits (if missing) + audio cache
  cv        5-fold CV on the dev set: one model per fold + out-of-fold (OOF) predictions
  evaluate  final evaluation on the held-out test set (--dev-only: CV results of an ensemble)
  predict   predictions for new recordings (bonus) -> CSV patient_id,murmur
  report    comparison table and figures of every experiment

All decisions (architecture, features, hyper-parameters, aggregation, decision rule)
are taken from the dev CV; the test set is evaluated once, at the end.

Examples:
  python -m murmur prepare
  python -m murmur cv --config configs/cnn.yaml
  python -m murmur cv --config configs/cnn.yaml --name cnn_lr3e-4 --set train.lr=3e-4
  python -m murmur evaluate --exp cnn rnn --dev-only
  python -m murmur report
  python -m murmur evaluate --exp cnn
  python -m murmur cv --config configs/cnn.yaml --all-data
  python -m murmur predict --exp cnn_all rnn_all --data-dir ../hidden-data --out predictions.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (CLASS_TO_IDX, CLASSES, WA_WEIGHTS, load_patients, load_recordings, load_signals, make_splits,
                   recording_labels)
from .evaluation import (PROB_COLS, bootstrap_ci, classification_metrics, decide, describe_decision, fit_decision,
                         patient_scores, weighted_accuracy)
from .models import build_model
from .plots import plot_confusion_matrix, plot_cv_comparison, plot_learning_curves
from .training import patient_wa, predict_recordings, train_model
from .utils import Logger, get_device, load_config, load_json, save_config, save_json, set_seed

DEFAULT_DEVICES = ("cuda", "mps", "cpu")


# ----------------------------------------------------------------------------- experiment helpers

def load_dataset(cfg):
    """Patients, recordings (with split columns and recording labels), patient labels and splits."""
    data_dir = cfg["data"]["dir"]
    patients = load_patients(data_dir)
    splits = pd.read_csv(cfg["data"]["splits"])
    recordings = load_recordings(data_dir).merge(
        splits[["patient_id", "subset", "fold", "fold_all"]], on="patient_id", how="inner")
    recordings["label"] = recording_labels(recordings, patients, cfg["data"]["label_mode"])
    patient_labels = patients.set_index("patient_id")["murmur"].map(CLASS_TO_IDX)
    return patients, recordings, patient_labels, splits


def window_params(cfg):
    fs = cfg["audio"]["fs"]
    return int(cfg["windows"]["length_s"] * fs), int(cfg["windows"]["eval_hop_s"] * fs)


def resolve(exp, output_dir):
    path = Path(exp)
    return path if path.is_dir() else Path(output_dir) / exp


def load_experiment(exp_dir):
    """Config, CV summary and out-of-fold recording predictions of a finished CV run."""
    exp_dir = Path(exp_dir)
    return (load_config([exp_dir / "config.yaml"]), load_json(exp_dir / "cv_summary.json"),
            pd.read_csv(exp_dir / "oof_recordings.csv"))


def combine_oof(oofs):
    """Average the out-of-fold probabilities of several experiments (same recordings and folds)."""
    base = oofs[0][["rec_id", "patient_id", "location", "fold"]].sort_values("rec_id").reset_index(drop=True)
    probs = []
    for oof in oofs:
        o = oof.set_index("rec_id").loc[base["rec_id"]]
        if not (o["fold"].to_numpy() == base["fold"].to_numpy()).all():
            raise ValueError("experiments were trained on different folds and cannot be combined")
        probs.append(o[PROB_COLS].to_numpy())
    base[PROB_COLS] = np.mean(probs, axis=0)
    return base


def decision_for(oof, patient_labels, decision_cfg):
    """Decision rule (aggregation + multipliers) fitted on out-of-fold predictions."""
    return fit_decision(oof, patient_labels, oof.groupby("patient_id")["fold"].first(),
                        decision_cfg.get("methods", ("max", "mean")), decision_cfg.get("tolerance", 0.005))


def cv_summary(name, cfg, oof, patient_labels, folds, all_data):
    decision = decision_for(oof, patient_labels, cfg["decision"])
    return {"experiment": name, "all_data": all_data, "folds": [int(k) for k in folds],
            "decision": {k: v for k, v in decision.items() if k != "methods"},
            "methods": decision["methods"]}, decision


def fold_models(exp_dir, device):
    """Yield (fold, model) for every fold model saved in an experiment directory."""
    for path in sorted(Path(exp_dir).glob("fold*/model.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        yield int(path.parent.name[4:]), model.to(device).eval()


def predict_with_experiment(exp_dir, cfg, signals_list, device):
    """Recording probabilities of every fold model of an experiment: array (n_folds, R, 3)."""
    length, hop = window_params(cfg)
    return np.stack([predict_recordings(m, signals_list, length, hop, device) for _, m in fold_models(exp_dir, device)])


def load_ensemble(exps, output_dir):
    """Directories, (config, summary, oof) of each experiment, and the decision rule of their ensemble."""
    exp_dirs = [resolve(e, output_dir) for e in exps]
    loaded = [load_experiment(d) for d in exp_dirs]
    cfg = loaded[0][0]
    _, recordings, patient_labels, _ = load_dataset(cfg)
    decision = decision_for(combine_oof([oof for _, _, oof in loaded]), patient_labels, cfg["decision"])
    return exp_dirs, loaded, recordings, patient_labels, decision


def describe(cfg):
    """Short description of the model and of its input, for tables and titles."""
    m, f = cfg["model"], cfg["features"]
    if m["type"] == "cnn":
        model = f"CNN {'-'.join(map(str, m.get('channels', [])))}"
    else:
        model = f"{'Bi' if m.get('bidirectional', True) else ''}{m.get('cell', 'gru').upper()} {m.get('layers', 2)}x{m.get('hidden', 64)}"
    if f.get("kind") == "logmel":
        feats = f"log-mel {f.get('n_mels', 64)}"
    else:
        feats = f"MFCC {f.get('n_mfcc', 20)}{'+d+dd' if f.get('deltas', True) else ''}"
    return model, f"{feats}, {cfg['windows']['length_s']:g}s"


def trivial_baselines(splits):
    """WA of always predicting Present / Absent on the dev patients."""
    counts = splits.loc[splits["subset"] == "dev", "murmur"].map(CLASS_TO_IDX).value_counts().reindex(range(3), fill_value=0)
    total = float((counts.to_numpy() * WA_WEIGHTS).sum())
    return {"always Present": 5 * counts[0] / total, "always Absent": counts[2] / total}


# ----------------------------------------------------------------------------- commands

def cmd_prepare(args):
    cfg = load_config(args.config, args.overrides)
    patients = load_patients(cfg["data"]["dir"])
    split_path = Path(cfg["data"]["splits"])
    if split_path.exists() and not args.force:
        print(f"using existing splits: {split_path} (--force to recreate)")
        splits = pd.read_csv(split_path)
    else:
        splits = make_splits(patients, args.n_folds, args.test_fraction, cfg["seed"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(split_path, index=False)
        print(f"splits written to {split_path}")

    n_shared = splits.groupby("subject_id")["patient_id"].transform("size").gt(1).sum()
    print(f"\n{len(splits)} patients, {splits['subject_id'].nunique()} subjects "
          f"({n_shared} patient ids belong to children recorded in both campaigns)")
    print("\npatients per subset:")
    print(pd.crosstab(splits["subset"], splits["murmur"], margins=True))
    print("\ndev folds:")
    print(pd.crosstab(splits.loc[splits["subset"] == "dev", "fold"], splits["murmur"], margins=True))
    assert splits.groupby("subject_id")["subset"].nunique().max() == 1, "a subject appears in both dev and test"

    recordings = load_recordings(cfg["data"]["dir"])
    if recordings.empty:
        print(f"\nno recordings found in {cfg['data']['dir']}: download the dataset (scripts/download_data.sh)")
        return
    print(f"\n{len(recordings)} recordings of {recordings['patient_id'].nunique()} patients found")
    load_signals(recordings, cfg["audio"], cfg.get("cache_dir"))
    print("audio cache ready")


def cmd_cv(args):
    cfg = load_config(args.config, args.overrides)
    name = args.name or cfg["name"]
    if args.all_data and not name.endswith("_all"):
        name += "_all"
    out = Path(cfg["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    log = Logger(out / "train.log")
    device = get_device(args.device, cfg.get("device_preference", DEFAULT_DEVICES))
    log(f"experiment {name} on {device}")

    _, recordings, patient_labels, _ = load_dataset(cfg)
    fold_col = "fold_all" if args.all_data else "fold"
    pool = recordings if args.all_data else recordings[recordings["subset"] == "dev"]
    signals = load_signals(recordings, cfg["audio"], cfg.get("cache_dir"))
    length, hop = window_params(cfg)

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
        fold_wa = patient_wa(va, oof[PROB_COLS].to_numpy(), patient_labels, cfg["decision"]["monitor"])
        log(f"fold {k} done in {(time.time() - t0) / 60:.1f} min (validation WA of the kept weights {fold_wa:.4f})")

    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(out / "oof_recordings.csv", index=False)
    summary, decision = cv_summary(name, cfg, oof, patient_labels, folds, args.all_data)
    save_json(summary, out / "cv_summary.json")
    log(f"\n=== {name}: patient-level weighted accuracy over {len(folds)} folds ===\n{describe_decision(decision)}")


def cmd_evaluate(args):
    exp_dirs, loaded, recordings, patient_labels, decision = load_ensemble(args.exp, args.output_dir)
    if any(summary["all_data"] for _, summary, _ in loaded):
        raise SystemExit("experiments trained with --all-data saw the test patients and cannot be evaluated on test")
    name = args.name or "+".join(d.name for d in exp_dirs)
    out = Path(args.output_dir) / name / ("dev" if args.dev_only else "test")
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name}: dev cross-validation ===\n{describe_decision(decision)}")
    save_json({"experiments": [str(d) for d in exp_dirs], **decision}, out / "decision.json")
    if args.dev_only:
        return

    cfg = loaded[0][0]
    device = get_device(args.device, cfg.get("device_preference", DEFAULT_DEVICES))
    test = recordings[recordings["subset"] == "test"].reset_index(drop=True)
    fold_probs = []  # (n_experiments, n_folds, R, 3)
    for (exp_cfg, _, _), d in zip(loaded, exp_dirs):
        signals = load_signals(recordings, exp_cfg["audio"], exp_cfg.get("cache_dir"))
        fold_probs.append(predict_with_experiment(d, exp_cfg, [signals[r] for r in test["rec_id"]], device))
    fold_probs = np.mean(fold_probs, axis=0)  # ensemble over experiments, per fold

    def patient_predictions(probs):
        rec = test[["rec_id", "patient_id", "location"]].copy()
        rec[PROB_COLS] = probs
        scores = patient_scores(rec, decision["method"])
        return rec, scores, decide(scores.to_numpy(), decision["scale"])

    rec, scores, pred = patient_predictions(fold_probs.mean(axis=0))  # ensemble of the fold models
    y = patient_labels.loc[scores.index].to_numpy()
    metrics = classification_metrics(y, pred)
    metrics["weighted_accuracy_95ci"] = bootstrap_ci(y, pred)
    single = [weighted_accuracy(y, patient_predictions(p)[2]) for p in fold_probs]
    metrics["single_fold_models_wa"] = single
    metrics["single_fold_models_wa_mean"], metrics["single_fold_models_wa_std"] = float(np.mean(single)), float(np.std(single))
    save_json({"experiments": [str(d) for d in exp_dirs], "n_patients": len(y), **metrics}, out / "test_results.json")

    pd.DataFrame({"patient_id": scores.index, "true": [CLASSES[i] for i in y], "pred": [CLASSES[i] for i in pred],
                  **{c: scores[c].to_numpy() for c in PROB_COLS}}).to_csv(out / "test_predictions.csv", index=False)
    rec.to_csv(out / "test_recordings.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], f"{name}: test (WA = {metrics['weighted_accuracy']:.3f})",
                          out / "confusion_matrix.png")

    lo, hi = metrics["weighted_accuracy_95ci"]
    print(f"\n=== {name}: test set ({len(y)} patients) ===")
    print(f"  weighted accuracy (ensemble of the {len(single)} fold models): "
          f"{metrics['weighted_accuracy']:.4f}  (95% bootstrap CI {lo:.3f}-{hi:.3f})")
    print(f"  single fold models: {np.mean(single):.4f} +/- {np.std(single):.4f}")
    print("  confusion matrix (rows = predicted P/U/A, columns = true P/U/A):")
    print(np.array(metrics["confusion_matrix"]))
    print("  additional: accuracy {accuracy:.3f}, macro-F1 {macro_f1:.3f}, recall P/U/A "
          "{recall_Present:.3f}/{recall_Unknown:.3f}/{recall_Absent:.3f}".format(**metrics))
    print(f"results saved in {out}")


def cmd_predict(args):
    exp_dirs, loaded, _, _, decision = load_ensemble(args.exp, args.output_dir)
    print(f"decision rule (fitted on the out-of-fold predictions):\n{describe_decision(decision)}")
    recordings = load_recordings(args.data_dir)
    if recordings.empty:
        raise SystemExit(f"no recordings found in {args.data_dir}")
    print(f"{len(recordings)} recordings of {recordings['patient_id'].nunique()} patients")

    cfg = loaded[0][0]
    device = get_device(args.device, cfg.get("device_preference", DEFAULT_DEVICES))
    probs = []
    for (exp_cfg, _, _), d in zip(loaded, exp_dirs):
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


def cmd_report(args):
    if args.exp:
        dirs = [resolve(e, args.output_dir) for e in args.exp]
    else:
        dirs = sorted(p.parent for p in Path(args.output_dir).glob("*/cv_summary.json"))
    if not dirs:
        raise SystemExit("no finished experiments found")
    out = Path(args.output_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = None
    for d in dirs:
        cfg, summary, oof = load_experiment(d)
        if labels is None:
            labels = load_dataset(cfg)[2]
        summary, _ = cv_summary(d.name, cfg, oof, labels, summary["folds"], summary["all_data"])
        save_json(summary, d / "cv_summary.json")  # refreshed with the current decision code
        dec = summary["decision"]
        argmax = summary["methods"][dec["method"]]["argmax"]
        model, features = describe(cfg)
        rows.append({
            "experiment": d.name, "model": model, "features": features,
            "labels": cfg["data"]["label_mode"], "loss weights": str(cfg["train"]["class_weights"]),
            "folds": len(summary["folds"]),
            "decision": f"{dec['method']} + {'tuned' if dec['tuned'] else 'argmax'}",
            "WA argmax": f"{argmax['wa_mean']:.3f} ± {argmax['wa_std']:.3f}",
            "WA (chosen rule)": f"{dec['wa_mean']:.3f} ± {dec['wa_std']:.3f}",
            "_mean": dec["wa_mean"], "_std": dec["wa_std"],
        })
        histories = [pd.read_csv(p) for p in sorted(d.glob("fold*/history.csv"))]
        if histories:
            plot_learning_curves(histories, f"{d.name}: {model}, {features}", out / f"{d.name}_learning_curves.png")
        plot_confusion_matrix(dec["confusion_matrix"], f"{d.name}: dev CV, out-of-fold (WA = {dec['pooled_wa']:.3f})",
                              out / f"{d.name}_cv_confusion.png")

    table = pd.DataFrame(rows).sort_values("_mean", ascending=False)
    cols = [c for c in table.columns if not c.startswith("_")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table[cols].itertuples(index=False)]
    (out / "cv_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    baselines = trivial_baselines(pd.read_csv(load_experiment(dirs[0])[0]["data"]["splits"]))
    plot_cv_comparison(table["experiment"].to_numpy(), table["_mean"].to_numpy(), table["_std"].to_numpy(),
                       out / "cv_comparison.png", {k: round(v, 3) for k, v in baselines.items()})
    print("\ntrivial baselines on dev: " + ", ".join(f"{k} {v:.3f}" for k, v in baselines.items()))
    print(f"figures and table saved in {out}")


# ----------------------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m murmur", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def config_args(p, default=None):
        p.add_argument("--config", nargs="+", default=default, required=default is None, help="YAML file(s), merged in order")
        p.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE", help="e.g. train.lr=3e-4")

    def experiment_args(p):
        p.add_argument("--exp", nargs="+", required=True, help="experiment names (or directories); several = ensemble")
        p.add_argument("--output-dir", default="outputs")
        p.add_argument("--device", default="auto")

    p = sub.add_parser("prepare", help="splits + audio cache")
    config_args(p, default=["configs/base.yaml"])
    p.add_argument("--force", action="store_true", help="recreate the splits even if the file exists")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("cv", help="cross-validation on the dev set")
    config_args(p)
    p.add_argument("--name", help="experiment name (default: `name` in the config)")
    p.add_argument("--folds", nargs="*", type=int, help="run only these folds")
    p.add_argument("--all-data", action="store_true",
                   help="CV over dev + test patients, to retrain the final model after the test evaluation")
    p.add_argument("--resume", action="store_true", help="skip folds that are already finished")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser("evaluate", help="final evaluation on the test set")
    experiment_args(p)
    p.add_argument("--name", help="result folder (default: experiment names joined by '+')")
    p.add_argument("--dev-only", action="store_true", help="only the CV results (does not touch the test set)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict new recordings (bonus)")
    experiment_args(p)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("report", help="comparison table and figures")
    p.add_argument("--exp", nargs="*", help="experiments (default: all in --output-dir)")
    p.add_argument("--output-dir", default="outputs")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)
