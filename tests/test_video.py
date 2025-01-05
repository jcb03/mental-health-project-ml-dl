import pytest
import numpy as np
from src.data.video_processor import VideoProcessor
from src.models.video_model import VideoModel
from src.config.config import Config

def test_video_processing():
    video_processor = VideoProcessor()
    sample_frames = np.random.randn(30, 224, 224, 3)  # 30 frames, 224x224 RGB
    embeddings = video_processor.extract_features(sample_frames)
    assert embeddings.shape[1] == 512, "Incorrect embedding size."

def test_video_model_training():
    video_model = VideoModel()
    X_train = np.random.randn(10, 512)  # 10 samples, 512-dimensional embeddings
    y_train = np.random.randint(0, 2, 10)  # Binary labels
    video_model.train(X_train, y_train, X_train, y_train)
    preds = video_model.predict(X_train)
    assert len(preds) == len(y_train), "Prediction length mismatch."
