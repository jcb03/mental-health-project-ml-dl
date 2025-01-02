import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_audio_data(directory):
    # Load audio features and labels
    data = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            label = file.split("_")[0]  # Assuming labels are in filenames
            data.append(file_path)
            labels.append(label)
    return data, labels

def load_text_data(file_path):
    df = pd.read_csv(file_path)
    return df["text"], df["label"]

def load_video_data(directory):
    # Load video frames and labels
    data = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(directory, file)
            label = file.split("_")[0]  # Assuming labels are in filenames
            data.append(file_path)
            labels.append(label)
    return data, labels

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
