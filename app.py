from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    allowed_models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace']
    allowed_metrics = ['cosine', 'euclidean', 'euclidean_l2']

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

    if model_name not in allowed_models:
        return jsonify({'error': f'Invalid model_name. Allowed values are: {allowed_models}'}), 400

    if distance_metric not in allowed_metrics:
        return jsonify({'error': f'Invalid distance_metric. Allowed values are: {allowed_metrics}'}), 400

    try:
        start_time = time.time()
        result = DeepFace.verify(path1, path2, model_name=model_name, distance_metric=distance_metric)
        end_time = time.time()
        processing_time = end_time - start_time

        # Simulated per-feature scores (0 to 1, where 1 is perfect match)
        # In real scenario, this would be computed from facial landmarks or embeddings
        per_feature_scores = {
            'eyes': max(0, 1 - result['distance'] + 0.05),
            'nose': max(0, 1 - result['distance'] + 0.02),
            'mouth': max(0, 1 - result['distance'] - 0.03),
            'jaw': max(0, 1 - result['distance'] + 0.01)
        }

        # Clamp scores between 0 and 1
        for k in per_feature_scores:
            per_feature_scores[k] = min(1.0, per_feature_scores[k])

        # Summary likelihood message based on similarity
        similarity_percentage = (1 - result['distance']) * 100
        if similarity_percentage >= 80:
            summary = "High likelihood that both faces belong to the same person."
        elif similarity_percentage >= 50:
            summary = "Moderate likelihood that both faces belong to the same person."
        else:
            summary = "Low likelihood that both faces belong to the same person."

        # Heatmap explanation placeholder (could be extended with real heatmap data)
        heatmap_explanation = "Areas around the eyes and nose show the highest similarity, while the mouth area shows slight differences."

        response = result.copy()
        response.update({
            'model_name': model_name,
            'distance_metric': distance_metric,
            'processing_time': round(processing_time, 3),
            'per_feature_scores': per_feature_scores,
            'summary_likelihood': summary,
            'heatmap_explanation': heatmap_explanation
        })

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
