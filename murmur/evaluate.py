"""Final evaluation on the held-out test set.

Run it only once the models are chosen: every decision must come from the dev CV
(murmur.cv). The decision rule is fitted on the out-of-fold predictions, never on test.

    python -m murmur.evaluate --exp cnn
    python -m murmur.evaluate --exp rnn
    python -m murmur.evaluate --exp cnn rnn --dev-only          # CV results of the CNN+RNN ensemble
    python -m murmur.evaluate --exp cnn rnn --name cnn+rnn      # test results of the ensemble
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import PROB_COLS, decide, describe_decision, patient_scores
from .audio import load_signals
from .data import CLASSES
from .experiment import combine_oof, decision_for, load_dataset, load_experiment, predict_with_experiment
from .metrics import bootstrap_ci, classification_metrics, weighted_accuracy
from .plots import plot_confusion_matrix
from .utils import get_device, save_json


def resolve(exp, output_dir):
    path = Path(exp)
    return path if path.is_dir() else Path(output_dir) / exp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", required=True, help="experiment names (or directories)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--name", help="name of the result folder (default: experiment names joined by '+')")
    parser.add_argument("--dev-only", action="store_true", help="only report the CV results (no test set)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    exp_dirs = [resolve(e, args.output_dir) for e in args.exp]
    exps = [load_experiment(d) for d in exp_dirs]
    if any(summary["all_data"] for _, summary, _ in exps):
        raise SystemExit("experiments trained with --all-data saw the test patients and cannot be evaluated on test")
    cfg = exps[0][0]
    _, recordings, patient_labels, _ = load_dataset(cfg)
    oof = combine_oof([o for _, _, o in exps])
    decision = decision_for(oof, patient_labels, cfg["decision"]["methods"])
    name = args.name or "+".join(d.name for d in exp_dirs)
    out = Path(args.output_dir) / name / ("dev" if args.dev_only else "test")
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== {name}: dev cross-validation ===\n{describe_decision(decision)}")
    save_json({"experiments": [str(d) for d in exp_dirs], **decision}, out / "decision.json")
    if args.dev_only:
        return

    device = get_device(args.device, cfg.get("device_preference", ("cuda", "mps", "cpu")))
    test = recordings[recordings["subset"] == "test"].reset_index(drop=True)
    fold_probs = []  # (n_experiments, n_folds, R, 3)
    for (exp_cfg, _, _), d in zip(exps, exp_dirs):
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
    print("  confusion matrix (rows = predicted P/U/A, cols = true P/U/A):")
    print(np.array(metrics["confusion_matrix"]))
    print("  additional: accuracy {accuracy:.3f}, macro-F1 {macro_f1:.3f}, recall P/U/A "
          "{recall_Present:.3f}/{recall_Unknown:.3f}/{recall_Absent:.3f}".format(**metrics))
    print(f"results saved in {out}")


if __name__ == "__main__":
    main()
