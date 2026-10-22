import numpy as np

from murmur.metrics import classification_metrics, confusion_matrix, weighted_accuracy


def official_weighted_accuracy(labels, outputs):
    """Re-implementation of compute_weighted_accuracy from physionetchallenges/evaluation-2022
    (one-hot labels/outputs, classes ordered Present, Unknown, Absent)."""
    weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    A = np.zeros((3, 3))
    for i in range(len(labels)):
        A[np.argmax(outputs[i]), np.argmax(labels[i])] += 1  # rows = outputs, columns = labels
    return np.trace(weights * A) / np.sum(weights * A)


def test_matches_formula_and_official_code():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    m = confusion_matrix(y, p)
    formula = (5 * m[0, 0] + 3 * m[1, 1] + m[2, 2]) / (5 * m[:, 0].sum() + 3 * m[:, 1].sum() + m[:, 2].sum())
    assert np.isclose(weighted_accuracy(y, p), formula)
    assert np.isclose(weighted_accuracy(y, p), official_weighted_accuracy(np.eye(3)[y], np.eye(3)[p]))


def test_trivial_classifiers_on_public_data():
    y = np.array([0] * 179 + [1] * 68 + [2] * 695)  # public CirCor: Present, Unknown, Absent
    assert np.isclose(weighted_accuracy(y, np.full_like(y, 2)), 695 / 1794)  # always Absent: 0.387
    assert np.isclose(weighted_accuracy(y, np.full_like(y, 0)), 895 / 1794)  # always Present: 0.499


def test_confusion_matrix_convention():
    m = confusion_matrix(np.array([0, 0, 2]), np.array([2, 0, 2]))  # rows = predicted, columns = true
    assert m[2, 0] == 1 and m[0, 0] == 1 and m[2, 2] == 1
    r = classification_metrics(np.array([0, 0, 2]), np.array([2, 0, 2]))
    assert np.isclose(r["recall_Present"], 0.5) and np.isclose(r["precision_Absent"], 0.5)
