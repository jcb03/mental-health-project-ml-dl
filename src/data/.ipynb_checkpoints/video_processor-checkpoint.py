from deepface import DeepFace

def extract_video_emotions(image_path):
    analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"])
    return analysis["dominant_emotion"]
