from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images required'}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    try:
        result = DeepFace.verify(path1, path2, model_name='ArcFace', distance_metric='cosine')
        return jsonify({'verified': result['verified'], 'distance': result['distance']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
