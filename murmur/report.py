"""Comparison table and figures of the CV experiments (for the slides and the experiment log).

    python -m murmur.report                    # every experiment in outputs/
    python -m murmur.report --exp cnn rnn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_TO_IDX, WA_WEIGHTS
from .evaluate import resolve
from .experiment import load_experiment
from .plots import plot_confusion_matrix, plot_cv_comparison, plot_learning_curves


def describe(cfg):
    m, f = cfg["model"], cfg["features"]
    if m["type"] == "cnn":
        model = f"CNN {'-'.join(map(str, m.get('channels', [])))}"
    else:
        model = f"{'Bi' if m.get('bidirectional', True) else ''}{m.get('cell', 'gru').upper()} {m.get('layers', 2)}x{m.get('hidden', 64)}"
    feats = f"log-mel {f.get('n_mels', 64)}" if f.get("kind") == "logmel" else f"MFCC {f.get('n_mfcc', 20)}{'+d+dd' if f.get('deltas', True) else ''}"
    return model, f"{feats}, {cfg['windows']['length_s']:g}s"


def trivial_baselines(splits):
    counts = splits.loc[splits["subset"] == "dev", "murmur"].map(CLASS_TO_IDX).value_counts().reindex(range(3), fill_value=0)
    total = float((counts.to_numpy() * WA_WEIGHTS).sum())
    return {"always Present": 5 * counts[0] / total, "always Absent": counts[2] / total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="*", help="experiments (default: all in --output-dir)")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)

    if args.exp:
        dirs = [resolve(e, args.output_dir) for e in args.exp]
    else:
        dirs = sorted(p.parent for p in Path(args.output_dir).glob("*/cv_summary.json"))
    if not dirs:
        raise SystemExit("no finished experiments found")
    out = Path(args.output_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in dirs:
        cfg, summary, _ = load_experiment(d)
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

    cfg0 = load_experiment(dirs[0])[0]
    baselines = trivial_baselines(pd.read_csv(cfg0["data"]["splits"]))
    plot_cv_comparison(table["experiment"].to_numpy(), table["_mean"].to_numpy(), table["_std"].to_numpy(),
                       out / "cv_comparison.png", {k: round(v, 3) for k, v in baselines.items()})
    print(f"\ntrivial baselines on dev: " + ", ".join(f"{k} {v:.3f}" for k, v in baselines.items()))
    print(f"figures and table saved in {out}")


if __name__ == "__main__":
    main()
