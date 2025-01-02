import os

class Config:
    # General configurations
    RANDOM_SEED = 42

    # Dataset paths
    AUDIO_DATA_PATH = os.path.join("data", "raw", "RAVDESS")
    TEXT_DATA_PATH = os.path.join("data", "raw", "reddit", "text_data.csv")
    VIDEO_DATA_PATH = os.path.join("data", "raw", "video")

    # Processed data paths
    PROCESSED_AUDIO_FEATURES_PATH = os.path.join("data", "processed", "audio_features.pkl")
    PROCESSED_TEXT_FEATURES_PATH = os.path.join("data", "processed", "text_features.pkl")
    PROCESSED_VIDEO_FEATURES_PATH = os.path.join("data", "processed", "video_features.pkl")

    # Model paths
    MODEL_SAVE_DIR = os.path.join("models", "saved_models")
    AUDIO_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "audio_model.pkl")
    TEXT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "text_model.pkl")
    VIDEO_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "video_model.pkl")
    FUSION_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fusion_model.pkl")

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Audio processing parameters
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_FRAME_LENGTH = 2048
    AUDIO_HOP_LENGTH = 512
    AUDIO_MFCC_FEATURES = 13

    # Text processing parameters
    TEXT_MAX_SEQUENCE_LENGTH = 512
    TEXT_VOCAB_SIZE = 20000
    TEXT_EMBEDDING_DIM = 300

    # Video processing parameters
    VIDEO_FRAME_RATE = 30
    VIDEO_FACE_MODEL = "VGG-Face"  # DeepFace supported models: VGG-Face, Facenet, OpenFace, etc.

    # Logging and outputs
    LOG_DIR = os.path.join("logs")
    OUTPUT_DIR = os.path.join("outputs")

# Create necessary directories if they do not exist
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
