from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# =========================
# Load Saved Files
# =========================
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# Prediction Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_choice = request.form["model"]

        input_data = []

        for feature in feature_columns:

            # Special handling for employment years
            if feature == "DAYS_EMPLOYED":
                years = float(request.form["employment_years"])
                days_employed = - (years * 365)
                input_data.append(days_employed)

            else:
                value = float(request.form.get(feature))
                input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)

        # Apply preprocessing
        input_array = imputer.transform(input_array)
        input_array = scaler.transform(input_array)

        # Model selection
        if model_choice == "RandomForest":
            prediction = rf_model.predict(input_array)[0]
        else:
            prediction = lr_model.predict(input_array)[0]

        result = "Loan Approved ✅" if prediction == 0 else "Loan Default Risk ⚠"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
