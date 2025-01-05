import pytest
import numpy as np
from src.models.fusion_model import FusionModel

def test_fusion_model_training():
    fusion_model = FusionModel()
    X_train = np.random.randn(10, 512)  # Mock combined input
    y_train = np.random.randint(0, 2, 10)  # Binary labels
    fusion_model.train(X_train, y_train, X_train, y_train)
    preds = fusion_model.predict(X_train)
    assert len(preds) == len(y_train), "Prediction length mismatch."
