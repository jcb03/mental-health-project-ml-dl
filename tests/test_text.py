import pytest
from src.data.text_processor import TextProcessor
from src.models.text_model import TextModel
from src.config.config import Config

def test_text_processing():
    text_processor = TextProcessor()
    sample_text = "This is a test sentence."
    features = text_processor.extract_features([sample_text])
    assert features.shape[1] == Config.TEXT_MAX_SEQUENCE_LENGTH, "Incorrect sequence length."

def test_text_model_training():
    text_model = TextModel()
    X_train = [[0] * Config.TEXT_MAX_SEQUENCE_LENGTH for _ in range(10)]  # Mock tokenized input
    y_train = [0, 1] * 5  # Alternating binary labels
    text_model.train(X_train, y_train, X_train, y_train)
    preds = text_model.predict(X_train)
    assert len(preds) == len(y_train), "Prediction length mismatch."
