from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return "Heart Disease Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Input features
        features = np.array([[
            data["age"],
            data["gender"],
            data["chestpain"],
            data["restingBP"],
            data["serumcholestrol"],
            data["fastingbloodsugar"],
            data["restingrelectro"],
            data["maxheartrate"],
            data["exerciseangia"],
            data["oldpeak"],
            data["slope"],
        ]])

        # Scale input
        scaled_data = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]

        # Correct mapping
        result = "High Risk" if prediction == 0 else "Low Risk"

        return jsonify({
            "prediction": result,
            "risk_probability": round(probs[0] * 100, 2),
            "safe_probability": round(probs[1] * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)