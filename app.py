from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return "Heart Disease API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

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

        # STEP 2: scale
        scaled_data = scaler.transform(features)

        # STEP 3: prediction
        prediction = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]

        classes = model.classes_

        # STEP 4: correct probability mapping
        high_risk_index = list(classes).index(1)  # assume 1 = disease

        risk_probability = round(probs[high_risk_index] * 100, 2)
        safe_probability = round((1 - probs[high_risk_index]) * 100, 2)

        # STEP 5: result label
        result = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            "prediction": result,
            "risk_probability": risk_probability,
            "safe_probability": safe_probability
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)