import pickle
import os
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "alpha.pkl"

# Load model safely
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        print(f"❌ Model load error: {e}")
        model = None
else:
    print("⚠️ Model file not found! Please upload 'alpha.pkl'.")


# Home Page (Input Form)
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("output.html", prediction="❌ Model Not Available. Upload 'alpha.pkl'.")

    try:
        # Extract form inputs safely
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

        # Convert to numpy array for prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return render_template("output.html", prediction=f"{prediction:.2f}")

    except Exception as e:
        return render_template("output.html", prediction=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
