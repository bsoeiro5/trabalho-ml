"""Prepare the data: subject-level stratified splits and (if the audio is available) the audio cache.

    python -m murmur.prepare --config configs/base.yaml
"""
import argparse
from pathlib import Path

import pandas as pd

from .audio import load_signals
from .config import load_config
from .data import load_patients, load_recordings
from .splits import make_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="recreate the splits even if the file exists")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
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
    leaks = splits.groupby("subject_id")["subset"].nunique().gt(1).sum()
    assert leaks == 0, "a subject appears in both dev and test"

    recordings = load_recordings(cfg["data"]["dir"])
    if len(recordings) == 0:
        print("\nno recordings found: download the dataset (scripts/download_data.sh) to build the audio cache")
        return
    print(f"\n{len(recordings)} recordings of {recordings['patient_id'].nunique()} patients found")
    load_signals(recordings, cfg["audio"], cfg.get("cache_dir"))
    print("audio cache ready")


if __name__ == "__main__":
    main()
