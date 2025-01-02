from sklearn.ensemble import VotingClassifier

class FusionModel:
    def __init__(self, audio_model, text_model, video_model):
        self.model = VotingClassifier(
            estimators=[
                ("audio", audio_model),
                ("text", text_model),
                ("video", video_model),
            ],
            voting="hard"
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
