import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline_XGB.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "features.json")


model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH) as f:
    FEATURES = json.load(f)

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing {missing}"}), 400

    df = pd.DataFrame([data])
    df = df[FEATURES]

    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_price": round(float(prediction),2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
