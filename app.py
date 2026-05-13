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
    return "Heart Disease Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():

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
        data["noofmajorvessels"]
    ]])

    scaled_data = scaler.transform(features)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    result = "High Risk" if prediction == 1 else "Low Risk"

    return jsonify({
        "prediction": result,
        "probability": round(probability * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)