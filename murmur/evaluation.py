"""Evaluation: the Challenge metric and the rule that turns recording predictions into one decision per patient."""
import numpy as np

from .data import CLASSES, WA_WEIGHTS

# ----------------------------------------------------------------------------- metrics (patient level)

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


# ----------------------------------------------------------------------------- recordings -> patient
#
# From recording-level probabilities to exactly one decision per patient.
#
# Decision rule. The denominator of the weighted accuracy only depends on the true
# labels, so maximising it means maximising sum_i w(y_i) [pred_i == y_i], whose Bayes
# rule is  pred = argmax_c w_c p(c | x).  Training with a cross-entropy weighted by
# w = (5, 3, 1) makes the network output q_c proportional to w_c p(c | x), so the plain
# argmax already approximates that rule. Because the patient score comes from an
# aggregation of several recordings (and the network is not perfectly calibrated), we
# also tune multipliers for the Present and Unknown scores on validation predictions.

PROB_COLS = ["p_present", "p_unknown", "p_absent"]
SCALE_GRID = 2.0 ** np.arange(-3, 3.01, 0.25)


def patient_scores(rec, method="max"):
    """Combine the recordings of each patient into one score per class.

    rec: DataFrame with columns patient_id + PROB_COLS (one row per recording).
    mean: average of the recording probabilities.
    max:  Present score = highest Present probability over the recordings (a murmur
          heard at a single location is enough); Unknown and Absent are averaged.
    Returns a DataFrame indexed by patient_id with the three class scores.
    """
    g = rec.groupby("patient_id")
    scores = g[PROB_COLS].mean()
    if method == "max":
        scores["p_present"] = g["p_present"].max()
    elif method != "mean":
        raise ValueError(f"unknown aggregation {method!r}")
    return scores


def decide(scores, scale=(1.0, 1.0, 1.0)):
    """Predicted class index = argmax_c scale_c * score_c."""
    return np.argmax(np.asarray(scores) * np.asarray(scale)[None, :], axis=1)


def tune_scale(scores, y, grid=SCALE_GRID):
    """Grid-search the Present / Unknown multipliers (Absent fixed at 1) that maximise the WA.

    The WA surface is flat and noisy on a few hundred patients, so it is smoothed with
    a 3x3 mean filter, and ties are broken towards (1, 1), i.e. the plain argmax.
    """
    wa = np.array([[weighted_accuracy(y, decide(scores, (sp, su, 1.0))) for su in grid] for sp in grid])
    n = len(grid)
    padded = np.pad(wa, 1, mode="edge")
    smooth = sum(padded[i:i + n, j:j + n] for i in range(3) for j in range(3)) / 9
    log_g = np.abs(np.log2(grid))
    smooth -= 1e-6 * (log_g[:, None] + log_g[None, :])
    i, j = np.unravel_index(np.argmax(smooth), smooth.shape)
    return np.array([grid[i], grid[j], 1.0])


def crossfit_decision(scores, y, folds):
    """Cross-validated estimate of the whole decision procedure.

    For each fold, the multipliers are tuned on the out-of-fold predictions of the
    other folds and applied to this fold, so no patient is used to tune its own
    decision. Returns per-fold WA (argmax and tuned) and the pooled predictions.
    """
    scores, y, folds = np.asarray(scores), np.asarray(y), np.asarray(folds)
    pred = {"argmax": decide(scores), "tuned": np.empty(len(y), dtype=int)}
    fold_ids = np.unique(folds)
    for k in fold_ids:
        te = folds == k
        scale = tune_scale(scores[~te], y[~te]) if len(fold_ids) > 1 else np.ones(3)
        pred["tuned"][te] = decide(scores[te], scale)
    per_fold = {rule: [weighted_accuracy(y[folds == k], p[folds == k]) for k in fold_ids] for rule, p in pred.items()}
    return {"folds": fold_ids.tolist(), "per_fold": per_fold, "pred": pred}


def fit_decision(rec, patient_labels, patient_folds, methods=("mean", "max")):
    """Choose the decision rule by cross-fitted CV WA, then fit it on all OOF predictions.

    Candidates: every aggregation method, with the plain argmax or with tuned
    multipliers. Tuning only pays off when it beats the argmax on held-out folds;
    ties go to the simpler rule (argmax first, then the first method).
    rec: out-of-fold recording predictions (patient_id + PROB_COLS).
    patient_labels / patient_folds: Series indexed by patient_id.
    """
    results = {}
    for method in methods:
        scores = patient_scores(rec, method)
        y = patient_labels.loc[scores.index].to_numpy()
        cf = crossfit_decision(scores.to_numpy(), y, patient_folds.loc[scores.index].to_numpy())
        r = {"scale": tune_scale(scores.to_numpy(), y).tolist()}
        for rule in ("argmax", "tuned"):
            wa = cf["per_fold"][rule]
            r[rule] = {"wa_per_fold": wa, "wa_mean": float(np.mean(wa)), "wa_std": float(np.std(wa)),
                       "pooled_wa": weighted_accuracy(y, cf["pred"][rule]),
                       "confusion_matrix": confusion_matrix(y, cf["pred"][rule]).tolist()}
        results[method] = r
    candidates = [(False, m) for m in methods] + [(True, m) for m in methods]  # in order of preference
    tuned, method = max(candidates, key=lambda c: results[c[1]]["tuned" if c[0] else "argmax"]["wa_mean"])
    chosen = results[method]["tuned" if tuned else "argmax"]
    return {"method": method, "tuned": tuned, "scale": results[method]["scale"] if tuned else [1.0, 1.0, 1.0],
            **chosen, "methods": results}


def describe_decision(decision):
    """Human-readable summary of the output of fit_decision."""
    lines = []
    for method, r in decision["methods"].items():
        a, t = r["argmax"], r["tuned"]
        lines.append(f"  {method:5s} argmax {a['wa_mean']:.4f} +/- {a['wa_std']:.4f}   "
                     f"tuned (cross-fitted) {t['wa_mean']:.4f} +/- {t['wa_std']:.4f}")
    rule = f"multipliers {np.round(decision['scale'], 3).tolist()}" if decision["tuned"] else "argmax"
    lines.append(f"  chosen: {decision['method']} aggregation + {rule} -> "
                 f"CV WA {decision['wa_mean']:.4f} +/- {decision['wa_std']:.4f}")
    lines.append("  pooled out-of-fold confusion matrix (rows = predicted P/U/A, columns = true P/U/A):")
    lines += ["    " + " ".join(f"{v:4d}" for v in row) for row in decision["confusion_matrix"]]
    return "\n".join(lines)
