import cv2
from deepface import DeepFace
import numpy as np
import os

class VideoProcessor:
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_embeddings(self, frames):
        embeddings = []
        for frame in frames:
            try:
                embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                embeddings.append(embedding[0]["embedding"])
            except:
                continue
        return np.array(embeddings)

    def process_video(self, video_path):
        frames = self.extract_frames(video_path)
        return self.extract_embeddings(frames)

if __name__ == "__main__":
    vp = VideoProcessor()
    videos = os.listdir("data/raw/video/")
    for video in videos:
        embeddings = vp.process_video(f"data/raw/video/{video}")
        np.save(f"data/processed/video/{video.split('.')[0]}.npy", embeddings)
