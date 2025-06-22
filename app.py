import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
from werkzeug.utils import secure_filename
import time
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('face_similarity_landing.html')

@app.route('/checker')
def checker():
    return render_template('index.html')

@app.route('/learn-more')
def learn_more():
    return render_template('learn_more.html')

# Helper function to convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    allowed_models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace']
    allowed_metrics = ['cosine', 'euclidean', 'euclidean_l2']
    allowed_detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images required'}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    model_name = request.form.get('model_name', 'ArcFace')
    distance_metric = request.form.get('distance_metric', 'cosine')
    detector_backend = request.form.get('detector_backend', 'opencv')

    print(f"Debug: Received model_name = {model_name}")
    print(f"Debug: Received distance_metric = {distance_metric}")
    print(f"Debug: Received detector_backend = {detector_backend}")

    if model_name not in allowed_models:
        return jsonify({'error': f'Invalid model_name. Allowed values are: {allowed_models}'}), 400

    if distance_metric not in allowed_metrics:
        return jsonify({'error': f'Invalid distance_metric. Allowed values are: {allowed_metrics}'}), 400

    if detector_backend not in allowed_detectors:
        return jsonify({'error': f'Invalid detector_backend. Allowed values are: {allowed_detectors}'}), 400

    try:
        start_time = time.time()
        result = DeepFace.verify(path1, path2, model_name=model_name, distance_metric=distance_metric, detector_backend=detector_backend)
        end_time = time.time()

        # Convert DeepFace result to native Python types
        result = convert_numpy_types(result)

        processing_time = end_time - start_time

        # Simulated per-feature scores
        distance = result.get('distance', 1.0)
        per_feature_scores = {
            'eyes': max(0, 1 - distance + 0.05),
            'nose': max(0, 1 - distance + 0.02),
            'mouth': max(0, 1 - distance - 0.03),
            'jaw': max(0, 1 - distance + 0.01)
        }

        # Clamp scores between 0 and 1
        for k in per_feature_scores:
            per_feature_scores[k] = min(1.0, per_feature_scores[k])

        similarity_percentage = (1 - distance) * 100
        if similarity_percentage >= 80:
            summary = "High likelihood that both faces belong to the same person."
        elif similarity_percentage >= 50:
            summary = "Moderate likelihood that both faces belong to the same person."
        else:
            summary = "Low likelihood that both faces belong to the same person."

        heatmap_explanation = "Areas around the eyes and nose show the highest similarity, while the mouth area shows slight differences."

        response = result.copy()
        response.update({
            'model_name': model_name,
            'distance_metric': distance_metric,
            'detector_backend': detector_backend,
            'processing_time': round(processing_time, 3),
            'per_feature_scores': per_feature_scores,
            'summary_likelihood': summary,
            'heatmap_explanation': heatmap_explanation
        })

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error during DeepFace verification:\n{traceback_str}")
        return jsonify({'error': str(e), 'traceback': traceback_str}), 500

if __name__ == '__main__':
    app.run(debug=True)
