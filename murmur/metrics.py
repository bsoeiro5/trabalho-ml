"""Evaluation metrics, computed at the patient level."""
import numpy as np

from .data import CLASSES, WA_WEIGHTS


def confusion_matrix(y_true, y_pred, n_classes=3):
    """m[i, j] = number of patients predicted as class i whose true class is j (Challenge convention)."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (np.asarray(y_pred), np.asarray(y_true)), 1)
    return m


def weighted_accuracy(y_true, y_pred, weights=WA_WEIGHTS):
    """Weighted accuracy of the PhysioNet Challenge 2022 murmur task.

    Accuracy where each patient counts with the weight of its true class
    (Present 5, Unknown 3, Absent 1), as in the official evaluation code.
    """
    m = confusion_matrix(y_true, y_pred, len(weights)) * np.asarray(weights)[None, :]
    total = m.sum()
    return float(np.trace(m) / total) if total > 0 else float("nan")


def _ratio(a, b):
    return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)


def classification_metrics(y_true, y_pred):
    """Weighted accuracy, confusion matrix and additional metrics (per-class recall, precision, F1)."""
    m = confusion_matrix(y_true, y_pred)
    tp = np.diag(m).astype(float)
    recall = _ratio(tp, m.sum(axis=0))
    precision = _ratio(tp, m.sum(axis=1))
    f1 = _ratio(2 * precision * recall, precision + recall)
    res = {
        "weighted_accuracy": weighted_accuracy(y_true, y_pred),
        "accuracy": float(tp.sum() / m.sum()),
        "macro_f1": float(np.nanmean(f1)),
        "confusion_matrix": m.tolist(),
    }
    for c, name in enumerate(CLASSES):
        res[f"recall_{name}"] = float(recall[c])
        res[f"precision_{name}"] = float(precision[c])
        res[f"f1_{name}"] = float(f1[c])
    return res


def bootstrap_ci(y_true, y_pred, metric=weighted_accuracy, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval, resampling patients with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, len(y_true), (n_boot, len(y_true)))]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
