import librosa
import numpy as np
import os

class AudioProcessor:
    def extract_mfcc(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)

    def process_audio_files(self, input_dir, output_dir):
        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                mfcc = self.extract_mfcc(os.path.join(input_dir, file))
                np.save(os.path.join(output_dir, f"{file.split('.')[0]}.npy"), mfcc)

if __name__ == "__main__":
    ap = AudioProcessor()
    ap.process_audio_files("data/raw/audio/", "data/processed/audio/")
