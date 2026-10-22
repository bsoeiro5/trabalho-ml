import numpy as np
import pandas as pd

from murmur.aggregation import crossfit_decision, decide, patient_scores, tune_scale


def test_patient_scores():
    rec = pd.DataFrame({"patient_id": [1, 1, 2], "p_present": [0.9, 0.1, 0.2],
                        "p_unknown": [0.0, 0.1, 0.1], "p_absent": [0.1, 0.8, 0.7]})
    mean, mx = patient_scores(rec, "mean"), patient_scores(rec, "max")
    assert np.isclose(mean.loc[1, "p_present"], 0.5) and np.isclose(mx.loc[1, "p_present"], 0.9)
    assert decide(mx.to_numpy()).tolist() == [0, 2]


def test_tuned_scale_prefers_argmax_when_it_is_optimal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    scores = np.eye(3)[y] * 0.8 + 0.1  # perfectly separable: argmax is optimal
    assert np.allclose(tune_scale(scores, y), [1, 1, 1])


def test_crossfit_improves_biased_scores():
    rng = np.random.default_rng(1)
    y = rng.choice(3, 600, p=[0.2, 0.1, 0.7])
    scores = rng.dirichlet(np.ones(3), 600) * 0.5 + np.eye(3)[y] * 0.5
    scores[:, 0] *= 0.3  # the model under-predicts Present
    res = crossfit_decision(scores, y, np.arange(600) % 5)
    assert np.mean(res["per_fold"]["tuned"]) > np.mean(res["per_fold"]["argmax"])
