"""Unit tests for src.detection.training.train_binary's pure training helper."""
import numpy as np

from src.detection.training import train_binary


def test_train_baseline_model_fits_and_predicts():
    X = np.array([[0.0], [0.0], [10.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    model = train_binary.train_baseline_model(X, y)
    assert list(model.predict([[0.0], [10.0]])) == [0, 1]
