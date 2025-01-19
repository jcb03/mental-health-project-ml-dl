from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from src.analyzer import Analyzer

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp4', 'jpg', 'png', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the Analyzer
analyzer = Analyzer()
analyzer.load_trained_models(
    "models/saved_models/audio_model.pkl",
    "models/saved_models/text_model.pkl",
    "models/saved_models/video_model.pkl",
    "models/saved_models/fusion_model.pkl",
)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Audio file
        audio_file = request.files.get('audio')
        audio_path = None
        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(audio_path)

        # Video file
        video_file = request.files.get('video')
        video_path = None
        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)

        # Text input
        text_input = request.form.get('text', '')

        # Check if at least one input is provided
        if not (audio_path or video_path or text_input):
            return jsonify({"error": "At least one input (audio, video, or text) is required!"}), 400

        # Make predictions
        prediction = analyzer.analyze(audio_path, text_input, video_path)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
