# src/data/audio_processor.py
pip install librosa
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

class RAVDESSProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        self.label_encoder = LabelEncoder()
    
    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1)
        ])
    
    def load_data(self):
        features = []
        labels = []
        
        for filename in os.listdir(self.data_path):
            if filename.endswith(".wav"):
                parts = filename.split('-')
                emotion = self.emotion_map[parts[2]]
                file_path = os.path.join(self.data_path, filename)
                
                try:
                    features.append(self.extract_features(file_path))
                    labels.append(emotion)
                except:
                    continue
        
        return np.array(features), self.label_encoder.fit_transform(labels)
