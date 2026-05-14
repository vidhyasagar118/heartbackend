from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# ==============================
# FLASK APP CONFIG
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODEL FILES
# ==============================
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))

except Exception as e:
    print("Error loading files:", e)
    model = None
    scaler = None
    feature_names = None


# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return jsonify({
        "message": "Heart Disease Prediction API Running"
    })


# ==============================
# PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model files not loaded"
        }), 500

    try:
        data = request.json

        # ==========================
        # GET INPUTS FROM FRONTEND
        # ==========================
        age = float(data["age"])
        gender = float(data["gender"])
        chestpain = float(data["chestpain"])
        restingBP = float(data["restingBP"])
        serumcholestrol = float(data["serumcholestrol"])
        fastingbloodsugar = float(data["fastingbloodsugar"])
        restingrelectro = float(data["restingrelectro"])
        maxheartrate = float(data["maxheartrate"])
        exerciseangia = float(data["exerciseangia"])
        oldpeak = float(data["oldpeak"])
        slope = float(data["slope"])

        # ==========================
        # CREATE INPUT DATAFRAME
        # ==========================
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_names
        )

        # ==========================
        # NUMERIC FEATURES
        # ==========================
        input_data["age"] = age
        input_data["gender"] = gender
        input_data["restingBP"] = restingBP
        input_data["serumcholestrol"] = serumcholestrol
        input_data["maxheartrate"] = maxheartrate
        input_data["oldpeak"] = oldpeak
        input_data["exerciseangia"] = exerciseangia
        input_data["fastingbloodsugar"] = fastingbloodsugar

        # ==========================
        # ONE HOT ENCODING
        # ==========================
        for col, val in zip(
            ["chestpain", "slope", "restingrelectro"],
            [chestpain, slope, restingrelectro]
        ):

            dummy_col = f"{col}_{float(val)}"

            if dummy_col in input_data.columns:
                input_data[dummy_col] = 1

        # ==========================
        # SCALE DATA
        # ==========================
        scaled_data = scaler.transform(input_data)

        # ==========================
        # PREDICTION
        # ==========================
        probability = model.predict_proba(scaled_data)[0][1] * 100

        # ==========================
        # RISK CATEGORY
        # ==========================
        if probability < 30:
            risk = "LOW RISK"

        elif probability < 70:
            risk = "MILD RISK"

        else:
            risk = "HIGH RISK"

        # ==========================
        # RETURN RESPONSE
        # ==========================
        return jsonify({
            "success": True,
            "risk_percentage": round(float(probability), 2),
            "risk_category": risk
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)