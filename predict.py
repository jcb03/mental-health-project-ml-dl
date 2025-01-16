import os
from src.analyzer import Analyzer

# Paths to saved models
AUDIO_MODEL_PATH = "models/saved_models/audio_model.pkl"
TEXT_MODEL_PATH = "models/saved_models/text_model.pkl"
VIDEO_MODEL_PATH = "models/saved_models/video_model.pkl"
FUSION_MODEL_PATH = "models/saved_models/fusion_model.pkl"

# Initialize the Analyzer
analyzer = Analyzer()
analyzer.load_trained_models(
    AUDIO_MODEL_PATH, TEXT_MODEL_PATH, VIDEO_MODEL_PATH, FUSION_MODEL_PATH
)

def predict(audio_path=None, text_input=None, video_path=None):
    # Ensure at least one input is provided
    if not (audio_path or text_input or video_path):
        raise ValueError("At least one input (audio, video, or text) is required!")

    # Perform analysis
    result = analyzer.analyze(audio_path, text_input, video_path)
    return result

if __name__ == "__main__":
    # Example inputs for testing
    audio_path = "data/raw/RAVDESS/sample.wav"
    text_input = "I feel very stressed and anxious about life."
    video_path = "data/raw/video/sample_frame.jpg"

    prediction = predict(audio_path, text_input, video_path)
    print("Prediction:", prediction)
