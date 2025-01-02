import cv2
from deepface import DeepFace

class VideoProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def capture_frame(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame
    
    def extract_features(self, frame):
        results = DeepFace.analyze(frame, actions=['emotion'])
        return results['emotion']
