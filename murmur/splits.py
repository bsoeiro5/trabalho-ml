"""Patient-level (in fact subject-level) stratified splits."""
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import subject_ids


def _group_folds(strata, groups, n_folds, seed):
    with warnings.catch_warnings():  # tiny strata (e.g. Unknown x CC2014) trigger a warning
        warnings.simplefilter("ignore", UserWarning)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [val for _, val in sgkf.split(np.zeros(len(strata)), strata, groups)]


def make_splits(patients, n_folds=5, test_fraction=0.2, seed=42):
    """Hold out a test set, then split the remaining (dev) patients into CV folds.

    Splits are grouped by subject (a child recorded in both campaigns has two patient
    ids) and stratified by murmur label x campaign. Columns of the result:
      subset    dev / test
      fold      CV fold of the dev patients (-1 for test patients)
      fold_all  folds over all patients, only used to retrain the final model on
                everything once the test evaluation is done
    """
    patients = patients.reset_index(drop=True)
    df = patients[["patient_id", "murmur", "campaign"]].copy()
    df["subject_id"] = subject_ids(patients)
    strata = (df["murmur"] + "|" + df["campaign"].fillna("")).to_numpy()
    groups = df["subject_id"].to_numpy()

    test_idx = _group_folds(strata, groups, int(round(1 / test_fraction)), seed)[0]
    df["subset"] = "dev"
    df.loc[test_idx, "subset"] = "test"

    df["fold"] = -1
    df["fold_all"] = -1
    for subset, columns in [("dev", ["fold", "fold_all"]), ("test", ["fold_all"])]:
        idx = np.flatnonzero(df["subset"] == subset)
        for k, val in enumerate(_group_folds(strata[idx], groups[idx], n_folds, seed)):
            df.loc[idx[val], columns] = k
    return df
