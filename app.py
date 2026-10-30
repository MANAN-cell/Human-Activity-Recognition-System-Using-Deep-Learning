
import os
import urllib.request
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "human_activity_model.keras"

MODEL_URL = "https://github.com/MANAN-cell/Human-Activity-Recognition-System-Using-Deep-Learning/releases/download/v1.0/human_activity_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading AI Model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model Downloaded Successfully!")

model = tf.keras.models.load_model(MODEL_PATH)

activities = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS"
]

@app.route("/")
def home():
    return jsonify({
        "message": "Human Activity Recognition API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        data = payload.get("features")

        if not isinstance(data, list) or len(data) != 561:
            return jsonify({"error": "Exactly 561 features required"}), 400

        features = np.asarray(data, dtype=np.float32)

        if not np.isfinite(features).all():
            return jsonify({"error": "Invalid feature values"}), 400

        features = features.reshape(1, 561, 1)
        prediction = model.predict(features, verbose=0)[0]
        index = int(np.argmax(prediction))

        return jsonify({
            "activity": activities[index],
            "confidence": round(float(prediction[index]) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
