from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# ✅ app pehle define karo
app = Flask(__name__)
CORS(app)

# ✅ model load
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ✅ home route
@app.route("/")
def home():
    return "Heart Disease Prediction API Running"

# ✅ predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        print("Incoming data:", data)  # debug

        features = np.array([[
            data.get("age", 0),
            data.get("gender", 0),
            data.get("chestpain", 0),
            data.get("restingBP", 0),
            data.get("serumcholestrol", 0),
            data.get("fastingbloodsugar", 0),
            data.get("restingrelectro", 0),
            data.get("maxheartrate", 0),
            data.get("exerciseangia", 0),
            data.get("oldpeak", 0),
            data.get("slope", 0),
            data.get("noofmajorvessels", 0)
        ]])

        scaled_data = scaler.transform(features)

        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        result = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            "prediction": result,
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ run server
if __name__ == "__main__":
    app.run(debug=True)