import pickle
import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = "alpha.pkl"

model, scaler = None, None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            loaded_data = pickle.load(file)

        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            model, scaler = loaded_data
        else:
            model = loaded_data
            print("⚠️ Scaler not found in pickle file, using default scaler!")
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
        print(f"❌ Model load error: {e}")
        model, scaler = None, None
else:
    print("⚠️ Model file not found! Please upload 'alpha.pkl'.")

if scaler is None:
    scaler = StandardScaler()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("output.html", prediction="❌ Model or Scaler Not Available. Upload 'alpha.pkl'.")

    try:
        features = []
        for key in ["Open", "High", "Low", "Close"]:
            value = request.form.get(key)
            if value is None or value.strip() == "":
                return render_template("output.html", prediction=f"⚠️ Missing or invalid input: {key}")
            try:
                value = float(value)
            except ValueError:
                return render_template("output.html", prediction=f"⚠️ Invalid number format: {key}")
            features.append(value)

        # Convert input to DataFrame
        input_df = pd.DataFrame([features], columns=["Open", "High", "Low", "Close"])

        # Ensure scaler is fitted before transforming
        if hasattr(scaler, 'mean_'):
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        # Model prediction
        prediction = model.predict(input_scaled)[0]

        # Add a random variation to the prediction
        import random
        random_factor = random.uniform(0.9, 1.1)  # Random multiplier between 0.9 and 1.1
        adjusted_prediction = prediction * random_factor

        return render_template("output.html", prediction=f"{adjusted_prediction:.2f}")
    except Exception as e:
        return render_template("output.html", prediction=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
