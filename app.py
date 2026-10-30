import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "human_activity_model.keras")

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
            return jsonify({
                "error": "Exactly 561 features are required"
            }), 400

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

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )