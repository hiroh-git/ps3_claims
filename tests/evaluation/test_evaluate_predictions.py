import numpy as np

from ps3.evaluation import evaluate_predictions


def test_perfect_predictions() -> None:
    y = np.array([0.5, 1.0, 2.0, 5.0])
    w = np.array([1.0, 0.5, 1.0, 2.0])

    metrics = evaluate_predictions(y, y, sample_weight=w)["value"]

    assert np.isclose(metrics["bias"], 0)
    assert np.isclose(metrics["mae"], 0)
    assert np.isclose(metrics["rmse"], 0)
