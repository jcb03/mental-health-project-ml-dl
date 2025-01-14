import os
from src.data.audio_processor import extract_audio_features
from src.data.text_processor import preprocess_text
from src.data.video_processor import extract_video_emotions
from src.models.audio_model import AudioModel
from src.models.text_model import TextModel
from src.models.video_model import VideoModel
from src.models.fusion_model import FusionModel
from src.utils.dataset import split_data
from src.utils.visualization import plot_training_history
import joblib

# Paths to datasets
AUDIO_DATA_PATH = "data/raw/RAVDESS/"
TEXT_DATA_PATH = "data/raw/reddit/text_data.csv"
VIDEO_DATA_PATH = "data/raw/video/"

# Paths to save models
MODEL_SAVE_PATH = "models/saved_models"

# Ensure model save path exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Train and save individual models
def train_audio_model():
    print("Training Audio Model...")
    audio_data, audio_labels = extract_audio_features(AUDIO_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(audio_data, audio_labels)
    audio_model = AudioModel()
    history = audio_model.train(X_train, y_train, X_test, y_test)
    joblib.dump(audio_model, os.path.join(MODEL_SAVE_PATH, "audio_model.pkl"))
    plot_training_history(history)

def train_text_model():
    print("Training Text Model...")
    text_data, text_labels = preprocess_text(TEXT_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(text_data, text_labels)
    text_model = TextModel()
    history = text_model.train(X_train, y_train, X_test, y_test)
    joblib.dump(text_model, os.path.join(MODEL_SAVE_PATH, "text_model.pkl"))
    plot_training_history(history)

def train_video_model():
    print("Training Video Model...")
    video_data, video_labels = extract_video_emotions(VIDEO_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(video_data, video_labels)
    video_model = VideoModel()
    history = video_model.train(X_train, y_train, X_test, y_test)
    joblib.dump(video_model, os.path.join(MODEL_SAVE_PATH, "video_model.pkl"))
    plot_training_history(history)

def train_fusion_model():
    print("Training Fusion Model...")
    # Load individual model predictions
    audio_predictions = joblib.load(os.path.join(MODEL_SAVE_PATH, "audio_model.pkl"))
    text_predictions = joblib.load(os.path.join(MODEL_SAVE_PATH, "text_model.pkl"))
    video_predictions = joblib.load(os.path.join(MODEL_SAVE_PATH, "video_model.pkl"))

    # Combine predictions for fusion
    X_fusion = [audio_predictions, text_predictions, video_predictions]
    y_fusion = ...  # Ground truth labels for fusion
    X_train, X_test, y_train, y_test = split_data(X_fusion, y_fusion)
    
    fusion_model = FusionModel()
    history = fusion_model.train(X_train, y_train, X_test, y_test)
    joblib.dump(fusion_model, os.path.join(MODEL_SAVE_PATH, "fusion_model.pkl"))
    plot_training_history(history)

if __name__ == "__main__":
    train_audio_model()
    train_text_model()
    train_video_model()
    train_fusion_model()
