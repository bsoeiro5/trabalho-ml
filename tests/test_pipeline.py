"""End-to-end test of every command on a small synthetic dataset (no real data needed, ~1 min)."""
import json
from pathlib import Path

import pandas as pd

from murmur.pipeline import main
from synthetic import make_dataset

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_pipeline(tmp_path):
    data = make_dataset(tmp_path / "data", n_patients=80, seed=0)
    hidden = make_dataset(tmp_path / "hidden", n_patients=8, seed=1)
    (hidden / "training_data.csv").unlink()  # hidden data: no labels
    out = tmp_path / "outputs"
    overrides = ["--set", f"data.dir={data}", f"data.splits={tmp_path / 'splits.csv'}", f"output_dir={out}",
                 f"cache_dir={tmp_path / 'cache'}", "train.epochs=2", "train.patience=2", "train.crops_per_recording=1"]

    main(["prepare", "--config", str(CONFIGS / "base.yaml"), *overrides])
    main(["cv", "--config", str(CONFIGS / "cnn.yaml"), *overrides])
    main(["cv", "--config", str(CONFIGS / "rnn.yaml"), *overrides])
    main(["evaluate", "--exp", "cnn", "rnn", "--output-dir", str(out), "--dev-only"])
    main(["evaluate", "--exp", "cnn", "--output-dir", str(out)])
    main(["report", "--output-dir", str(out)])
    main(["predict", "--exp", "cnn", "rnn", "--output-dir", str(out), "--data-dir", str(hidden),
          "--out", str(tmp_path / "predictions.csv")])

    summary = json.loads((out / "cnn" / "cv_summary.json").read_text())
    assert len(summary["folds"]) == 5 and 0 <= summary["decision"]["wa_mean"] <= 1
    results = json.loads((out / "cnn" / "test" / "test_results.json").read_text())
    assert sum(map(sum, results["confusion_matrix"])) == results["n_patients"]
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert list(pred.columns) == ["patient_id", "murmur"] and len(pred) == 8
    assert set(pred["murmur"]) <= {"Present", "Unknown", "Absent"}
