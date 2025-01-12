from src.models.audio_model import AudioModel
from src.models.text_model import TextModel
from src.models.video_model import VideoModel
from src.models.fusion_model import FusionModel
from src.data.audio_processor import extract_audio_features
from src.data.text_processor import preprocess_text
from src.data.video_processor import extract_video_emotions

class Analyzer:
    def __init__(self):
        # Initialize all models
        self.audio_model = AudioModel()
        self.text_model = TextModel()
        self.video_model = VideoModel()
        self.fusion_model = None  # Initialize later after training

    def load_trained_models(self, audio_path, text_path, video_path, fusion_path):
        self.audio_model = joblib.load(audio_path)
        self.text_model = joblib.load(text_path)
        self.video_model = joblib.load(video_path)
        self.fusion_model = joblib.load(fusion_path)

    def analyze(self, audio_path, text_input, video_frame_path):
        # Process inputs
        audio_features = extract_audio_features(audio_path)
        text_features = preprocess_text(text_input)
        video_emotion = extract_video_emotions(video_frame_path)

        # Make predictions
        audio_prediction = self.audio_model.predict([audio_features])
        text_prediction = self.text_model.predict([text_features])
        video_prediction = self.video_model.predict([video_emotion])

        # Combine predictions using fusion model
        final_prediction = self.fusion_model.predict(
            [[audio_prediction, text_prediction, video_prediction]]
        )
        return final_prediction
