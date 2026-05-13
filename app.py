from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return "Heart Disease Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input into array
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

        # Scale data
        scaled_data = scaler.transform(features)

        # Prediction
        prediction = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]

        # 🔥 ALWAYS CORRECT MAPPING
        result = "High Risk" if prediction == 0 else "Low Risk"
        probability = probs[prediction]

        return jsonify({
            "prediction": result,
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)