from deepface import DeepFace

def extract_video_emotions(image_path):
    analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
    return analysis['dominant_emotion']
