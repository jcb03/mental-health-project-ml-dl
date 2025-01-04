import pytest
import numpy as np
from src.data.audio_processor import AudioProcessor
from src.models.audio_model import AudioModel
from src.config.config import Config

def test_audio_processing():
    audio_processor = AudioProcessor()
    sample_audio = np.random.randn(22050)  # Simulate a 1-second random audio signal
    features = audio_processor.extract_features(sample_audio)
    assert features.shape[1] == Config.AUDIO_MFCC_FEATURES, "Incorrect number of MFCC features."

def test_audio_model_training():
    audio_model = AudioModel()
    X_train = np.random.randn(10, 40)  # 10 samples, 40 features (e.g., MFCCs)
    y_train = np.random.randint(0, 2, 10)  # Binary labels
    audio_model.train(X_train, y_train, X_train, y_train)
    preds = audio_model.predict(X_train)
    assert len(preds) == len(y_train), "Prediction length mismatch."
